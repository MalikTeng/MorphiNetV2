import os, sys, json, glob, tqdm
from collections import OrderedDict
from itertools import chain
from trimesh import Trimesh, load
from trimesh.convex import convex_hull
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import save_obj
from pytorch3d.ops import sample_points_from_meshes, taubin_smoothing
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops.marching_cubes import marching_cubes
from monai.data import DataLoader
from monai.losses import DiceCELoss, MaskedDiceLoss
from monai.metrics import DiceMetric, MSEMetric
from monai.networks.nets import DynUNet, SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, 
    AsDiscrete,
    KeepLargestConnectedComponent,
    RemoveSmallObjects,
    CropForegroundd,
    Resized,
    Spacingd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    EnsureTyped, 
)
from monai.transforms.utils import distance_transform_edt, generate_spatial_bounding_box
from monai.utils import set_determinism
# from einops import rearrange
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
import wandb
import plotly.figure_factory as ff

from data import *
from utils import *
from model.networks import *

from utils.rasterize.rasterize import Rasterize


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TrainPipeline:
    def __init__(
            self,
            super_params,
            seed, num_workers,
            is_training=True,
            **kwargs
        ):
        """
        :param 
            super_params: parameters for setting up dataset, network structure, training, etc.
            seed: random seed to shuffle data during augmentation.
            num_workers: tourch.utils.data.DataLoader num_workers.
            is_training: switcher for training (True, default) or testing (False).
        """
            
        self.super_params = super_params
        self.seed = seed
        self.num_workers = num_workers
        self.is_training = is_training
        self.target = kwargs.get("target")
        set_determinism(seed=self.seed)

        if is_training:
            self.ckpt_dir = os.path.join(super_params.ckpt_dir, "dynamic", super_params.run_id)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.unet_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "seg"]}
            )
            self.resnet_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "df"]}
            )
            self.gsn_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "chmf", "smooth"]}
            )
            self.ndf_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "ndf"]}
            )
            self.eval_df_score = OrderedDict(
                {k: np.asarray([]) for k in ["myo"]}
            )
            self.eval_msh_score = self.eval_df_score.copy()
            self.best_eval_score = 0
        else:
            self.ckpt_dir = super_params.ckpt_dir
            self.out_dir = super_params.out_dir
            os.makedirs(self.out_dir, exist_ok=True)

        # data augmentation for resizing the segmentation prediction into crop window size
        self.pred_transform = Compose([
            AsDiscrete(argmax=True),
            KeepLargestConnectedComponent(independent=True),
            RemoveSmallObjects(min_size=8),
        ])
        self.post_transform = Compose([
            Spacingd(["pred", "label"], [2.0, 2.0, 2.0], mode=("bilinear", "nearest"), allow_missing_keys=True),
            CropForegroundd(["pred", "label"], source_key="label", allow_missing_keys=True),
            Maskd(["pred", "label", "modal"], allow_missing_keys=True),
            FlexResized(
                ["pred", "label"], 
                (-1, self.super_params.crop_window_size[0], -1), 
                allow_missing_keys=True
                ),
            Resized(
                ["pred", "label"], 
                self.super_params.crop_window_size[0], 
                size_mode="longest", mode=("bilinear", "nearest-exact"), 
                allow_missing_keys=True
                ),
            ResizeWithPadOrCropd(
                ["pred", "label"],
                self.super_params.crop_window_size,
                mode="constant", value=0,
                allow_missing_keys=True
                ),

            EnsureTyped(["pred", "label"], device=DEVICE, allow_missing_keys=True),
        ])

        if super_params.use_ckpt is None:
            self._data_warper(rotation=True)

        # import control mesh (NDC space, [-1, 1]) to compute the subdivision matrix
        template_mesh = load(super_params.template_mesh_dir)
        centroid = template_mesh.bounds.mean(axis=0)
        extent = template_mesh.bounds.ptp(axis=0)
        template_mesh.apply_translation(-centroid)
        template_mesh.apply_scale(2 / extent)
        self._mesh_label(template_mesh)
        self.template_mesh = Meshes(
            verts=[torch.tensor(template_mesh.vertices, dtype=torch.float32)], 
            faces=[torch.tensor(template_mesh.faces, dtype=torch.int64)]
            ).to(DEVICE)

        self._prepare_modules()
        self._prepare_optimiser()

        self.rasterizer = Rasterize(self.super_params.crop_window_size) # tool for rasterizing mesh


    def _mesh_label(self, mesh):
        COLOR_MAPPING = {
            (1, 0, 0): 0,   # LV-ENDO
            (0, 1, 0): 1,   # RV-ENDO
            (0, 0, 1): 2,   # LV-EPI
            (1, 1, 0): 3,   # RV-EPI
            (1, 0, 1): 4,   # MV & AAV
            (0, 1, 1): 5,   # TV
            (1, 1, 1): 6,   # PV
            (0, 0, 0): 7,   # LEAVE OUT
        }
        vert_label = mesh.visual.vertex_colors[:, :3]
        vert_label = np.where(vert_label <= 85, 0, 1)
        vert_label = np.array([COLOR_MAPPING[tuple(c)] for c in vert_label])
        self.vert_label = torch.tensor(vert_label, dtype=torch.long, device=DEVICE)
        mesh_lv = convex_hull(mesh.vertices[np.any(np.stack([vert_label == i for i in [0]]), axis=0)])
        mesh_rv = convex_hull(mesh.vertices[np.any(np.stack([vert_label == i for i in [1]]), axis=0)])
        self.mesh_c = torch.tensor([mesh_lv.center_mass, mesh_rv.center_mass], device=DEVICE)


    def _data_warper(self, rotation:bool):
        
        if self.is_training or self.super_params.save_on == "cap":
            print(f"Preparing MR data {'with' if rotation else 'without'} rotation...")
            with open(self.super_params.mr_json_dir, "r") as f:
                mr_train_transform, mr_valid_transform = self._prepare_transform(
                    ["mr_image", "mr_label"], "mr", rotation, target=self.target
                    )
                mr_train_ds, mr_valid_ds, mr_test_ds = self._prepare_dataset(
                    json.load(f), "mr", mr_train_transform, mr_valid_transform
                )
                self.mr_train_loader, self.mr_valid_loader, self.mr_test_loader = self._prepare_dataloader(
                    mr_train_ds, mr_valid_ds, mr_test_ds
                )

        if (self.is_training or self.super_params.save_on == "sct") and not self.super_params._4d:
            print(f"Preparing CT data {'with' if rotation else 'without'} rotation...")
            with open(self.super_params.ct_json_dir, "r") as f:
                ct_train_transform, ct_valid_transform = self._prepare_transform(
                    ["ct_image", "ct_label"], "ct", rotation
                    )
                ct_train_ds, ct_valid_ds, ct_test_ds = self._prepare_dataset(
                    json.load(f), "ct", ct_train_transform, ct_valid_transform
                )
                self.ct_train_loader, self.ct_valid_loader, self.ct_test_loader = self._prepare_dataloader(
                    ct_train_ds, ct_valid_ds, ct_test_ds
                )


    def _prepare_transform(self, keys, modal, rotation, **kwargs):
        train_transform = pre_transform(
            keys, modal, "train", rotation,
            self.super_params.crop_window_size,
            self.super_params.pixdim, **kwargs
            )
        valid_transform = pre_transform(
            keys, modal, "valid", rotation,
            self.super_params.crop_window_size,
            self.super_params.pixdim, **kwargs
            )
        
        return train_transform, valid_transform


    def _remap_abs_path(self, data_list, modal, phase):
        if modal == "mr":
            return [{
                "mr_image": os.path.join(self.super_params.mr_data_dir, f"images{phase}", os.path.basename(d["image"])),
                "mr_label": os.path.join(self.super_params.mr_data_dir, f"labels{phase}", os.path.basename(d["label"])),
            } for d in data_list]
        elif modal == "ct":
            return [{
                "ct_image": os.path.join(self.super_params.ct_data_dir, f"images{phase}", os.path.split(d["image"])[-1]),
                "ct_label": os.path.join(self.super_params.ct_data_dir, f"labels{phase}", os.path.split(d["label"])[-1]),
            } for d in data_list]
        

    def _prepare_dataset(self, data_json, modal, train_transform, valid_transform):
        train_data = self._remap_abs_path(data_json["train_fold0"], modal, "Tr")
        valid_data = self._remap_abs_path(data_json["validation_fold0"], modal, "Tr")
        test_data = self._remap_abs_path(data_json["test"], modal, "Ts")

        if modal == "ct":
            train_data = train_data[:np.floor(self.super_params.ct_ratio * len(train_data)).astype(int)]

        if not self.is_training:
            train_ds = None
            valid_ds = None
            test_ds = Dataset(
                test_data, valid_transform, self.seed, sys.maxsize,
                self.super_params.cache_rate, self.num_workers
                )
        else:
            train_ds = Dataset(
                train_data, train_transform, self.seed, sys.maxsize,
                self.super_params.cache_rate, self.num_workers
                )
            valid_ds = Dataset(
                valid_data, valid_transform, self.seed, sys.maxsize,
                self.super_params.cache_rate, self.num_workers
                )
            test_ds = None
        
        return train_ds, valid_ds, test_ds


    def _prepare_dataloader(self, train_ds, valid_ds, test_ds):
        if not train_ds is None and train_ds.__len__() > 0:
            train_loader = DataLoader(
                train_ds, batch_size=self.super_params.batch_size,
                shuffle=True, num_workers=self.num_workers,
                collate_fn=collate_4D_batch,
                )
        else:
            train_loader = None
        if not valid_ds is None and valid_ds.__len__() > 0:
            val_loader = DataLoader(
                valid_ds, batch_size=1,
                shuffle=False, num_workers=self.num_workers,
                collate_fn=collate_4D_batch,
                )
        else:
            val_loader = None
        if not test_ds is None and test_ds.__len__() > 0:
            test_loader = DataLoader(
                test_ds, batch_size=1,
                shuffle=False, num_workers=self.num_workers,
                collate_fn=collate_4D_batch,
                )
        else:
            test_loader = None
        
        return train_loader, val_loader, test_loader


    def _prepare_modules(self):
        # initialise the df-predict module
        self.encoder_mr = DynUNet(
            spatial_dims=2, in_channels=1,
            out_channels=self.super_params.num_classes,
            kernel_size=self.super_params.kernel_size, 
            strides=self.super_params.strides,
            upsample_kernel_size=self.super_params.strides[1:], 
            filters=self.super_params.filters, 
            dropout=False,
            deep_supervision=False,
            res_block=True
        ).to(DEVICE)
        self.encoder_ct = DynUNet(
            spatial_dims=3, in_channels=1,
            out_channels=self.super_params.num_classes,
            kernel_size=self.super_params.kernel_size, 
            strides=self.super_params.strides,
            upsample_kernel_size=self.super_params.strides[1:], 
            filters=self.super_params.filters, 
            dropout=False,
            deep_supervision=False,
            res_block=True
        ).to(DEVICE)
        self.decoder = SegResNet(
            in_channels=self.super_params.num_classes,
            out_channels=self.super_params.num_classes,
            blocks_down=self.super_params.layers,
            blocks_up=tuple([1 for _ in range(len(self.super_params.layers)-1)]),
        ).to(DEVICE)

        # initialise the subdiv module
        self.subdivided_faces = Subdivision(self.template_mesh, self.super_params.subdiv_levels, mesh_label=self.vert_label) # create pre-computed subdivision matrix
        self.GSN = GSN(
            hidden_features=self.super_params.hidden_features_gsn, 
            num_layers=self.super_params.subdiv_levels if self.super_params.subdiv_levels > 0 else 2,
        ).to(DEVICE)

        # initialise th NDF module
        self.NDF = NODEBlock(
            hidden_size=16, atol=1, rtol=1e-2,
        ).to(DEVICE)


    def _prepare_optimiser(self):
        self.dice_loss_fn = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            )
        self.mse_loss_fn = nn.MSELoss()

        self.msk_dice_loss_fn = MaskedDiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            )
        self.l1_loss_fn = nn.L1Loss()

        # initialise the optimiser for unet
        self.optimzer_mr_unet = torch.optim.Adam(
            self.encoder_mr.parameters(), lr=self.super_params.lr
            )
        self.lr_scheduler_mr_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimzer_mr_unet, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
        self.optimzer_ct_unet = torch.optim.AdamW(
            self.encoder_ct.parameters(), lr=self.super_params.lr
            )
        self.lr_scheduler_ct_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimzer_ct_unet, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for resnet
        self.optimizer_resnet = torch.optim.AdamW(
            self.decoder.parameters(), 
            lr=self.super_params.lr
            )
        self.lr_scheduler_resnet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_resnet, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for gsn
        self.optimizer_gsn = torch.optim.AdamW(
            self.GSN.parameters(), 
            lr=self.super_params.lr
            )
        self.lr_scheduler_gsn = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gsn, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for ndf
        self.optimizer_ndf = torch.optim.Adam(
            self.NDF.parameters(),
            lr=self.super_params.lr
        )
        self.lr_scheduler_ndf = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_ndf, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel",
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
        )

        # initialise the gradient scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        torch.backends.cudnn.enabled = torch.backends.cudnn.is_available()
        torch.backends.cudnn.benchmark = torch.backends.cudnn.is_available()


    def surface_extractor(self, seg_true):
        """
            WARNING: this operation is non-differentiable.
            input:
                seg_true: ground truth segmentation.
            return:
                surface mesh with vertices and faces in NDC space [-1, 1].
        """
        seg_true_multi = [torch.any(torch.stack([seg_true == i for i in seg_idx]), dim=0) for seg_idx in [[1], [3], [2]]]   # lv, rv, myo

        mesh_true = []
        for seg_true_ in seg_true_multi:
            verts, faces = marching_cubes(
                seg_true_.squeeze(1).permute(0, 3, 1, 2).to(torch.float32), 
                isolevel=0.1,
                return_local_coords=True,
            )
            mesh_true.append(taubin_smoothing(Meshes(verts, faces), 0.77, -0.34, 30))

        return mesh_true


    @torch.no_grad()
    def warp_template_mesh(self, df_preds):
        """
            input:
                df preds: the predicted df.
            return:
                warped control mesh with vertices and faces in NDC space.
        """

        b, *_, d = df_preds.shape

        # def find_optimal_clusters_batch(points_batch, max_clusters=3):
        #     batch_size = points_batch.shape[0]
        #     optimal_clusters = torch.zeros(batch_size, dtype=torch.int64, device=points_batch.device)
        #     kmeans_results = []
            
        #     for i in range(batch_size):
        #         points = points_batch[i].cpu().numpy()
        #         silhouette_scores = []
        #         kmeans_models = []
        #         for n_clusters in range(2, max_clusters + 1):
        #             kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        #             cluster_labels = kmeans.fit_predict(points)
        #             silhouette_avg = silhouette_score(points, cluster_labels)
        #             silhouette_scores.append(silhouette_avg)
        #             kmeans_models.append(kmeans)
                
        #         best_index = silhouette_scores.index(max(silhouette_scores))
        #         optimal_clusters[i] = best_index + 2
        #         kmeans_results.append(kmeans_models[best_index])
            
        #     return optimal_clusters, kmeans_results

        # def find_cluster_centers_and_normals_batch(point_clouds, max_clusters=3):
        #     b, num_points, _ = point_clouds.shape

        #     # Find the optimal number of clusters and KMeans results for each point cloud in the batch
        #     n_clusters_batch, kmeans_results = find_optimal_clusters_batch(point_clouds, max_clusters)

        #     max_n_clusters = n_clusters_batch.max().item()

        #     # Initialize tensors to store cluster centers and normals
        #     centers = torch.full((b, max_n_clusters, 3), float('nan'), device=DEVICE)
        #     normals = torch.full((b, max_n_clusters, 3), float('nan'), device=DEVICE)

        #     # Process all point clouds in parallel
        #     for i in range(b):
        #         kmeans = kmeans_results[i]
        #         centers[i, :n_clusters_batch[i]] = torch.tensor(kmeans.cluster_centers_, device=DEVICE)
                
        #         # Get cluster labels for all points
        #         labels = torch.tensor(kmeans.labels_, dtype=torch.long, device=DEVICE)
                
        #         # Compute centered points for all clusters at once
        #         centered_points = point_clouds[i] - centers[i, labels]
                
        #         # Compute covariance matrices for all clusters
        #         cov_matrices = torch.zeros(n_clusters_batch[i], 3, 3, device=DEVICE)
        #         cov_matrices.index_add_(0, labels, centered_points.unsqueeze(2) * centered_points.unsqueeze(1))
                
        #         # Compute eigenvectors for all covariance matrices
        #         _, eigenvectors = torch.linalg.eigh(cov_matrices)
                
        #         # The eigenvector corresponding to the smallest eigenvalue is the normal
        #         cluster_normals = eigenvectors[:, :, 0]
                
        #         # Ensure normals point outward
        #         mean_centered_points = torch.zeros(n_clusters_batch[i], 3, device=DEVICE)
        #         mean_centered_points.index_add_(0, labels, centered_points)
        #         count = torch.bincount(labels, minlength=n_clusters_batch[i]).float().unsqueeze(1)
        #         mean_centered_points /= count
                
        #         dot_products = (cluster_normals * mean_centered_points).sum(dim=1)
        #         cluster_normals[dot_products < 0] *= -1
                
        #         normals[i, :n_clusters_batch[i]] = cluster_normals

        #     return centers, normals, n_clusters_batch

        # def find_cluster_normal(point_clouds, cluster_centers):
        #     # Compute centered points
        #     centered_points = point_clouds - cluster_centers.unsqueeze(1)

        #     # Compute covariance matrices
        #     cov_matrices = torch.bmm(centered_points.transpose(1, 2), centered_points).to(torch.float32)

        #     # Compute eigenvectors for all covariance matrices
        #     _, eigenvectors = torch.linalg.eigh(cov_matrices)

        #     # The eigenvector corresponding to the smallest eigenvalue is the normal
        #     normals = eigenvectors[:, :, 0]

        #     # Ensure normals point outward
        #     mean_centered_points = centered_points.mean(dim=1)
        #     dot_products = torch.sum(normals * mean_centered_points, dim=-1)
        #     normals[dot_products < 0] *= -1

        #     return normals
            
        # def find_closest_points(points, targets):
        #     distances = torch.cdist(targets.unsqueeze(1), points)
        #     closest_indices = distances.argmin(dim=2).squeeze(1)
        #     return closest_indices

        def find_rotation_matrix_xz(vector_msh, vector_df):
            # Project vectors onto xz-plane
            vector_msh_xz = torch.stack([vector_msh[:, 0], vector_msh[:, 2]], dim=1)
            vector_df_xz = torch.stack([vector_df[:, 0], vector_df[:, 2]], dim=1)

            # Normalize the projected vectors
            vector_msh_xz = vector_msh_xz / torch.norm(vector_msh_xz, dim=1, keepdim=True)
            vector_df_xz = vector_df_xz / torch.norm(vector_df_xz, dim=1, keepdim=True)

            # Calculate the cosine of the angle between the projected vectors
            cos_theta = torch.sum(vector_msh_xz * vector_df_xz, dim=1)

            # Calculate the sine of the angle using the determinant of 2x2 matrix
            sin_theta = vector_msh_xz[:, 0] * vector_df_xz[:, 1] - vector_msh_xz[:, 1] * vector_df_xz[:, 0]

            # Create rotation matrices
            R = torch.zeros(vector_msh.shape[0], 3, 3, device=vector_msh.device)
            R[:, 0, 0] = cos_theta
            R[:, 0, 2] = sin_theta
            R[:, 1, 1] = 1
            R[:, 2, 0] = -sin_theta
            R[:, 2, 2] = cos_theta

            return R

        # def find_rotation_matrix_rodrigues(source_norm, target_norm):
        #     """
        #     Find the rotation matrix that rotates source_norm to target_norm.
        #     If the angle is > 90 degrees, it aligns them on the same line.
            
        #     Args:
        #     source_norm (torch.Tensor): Source normal vectors of shape (batch_size, 3)
        #     target_norm (torch.Tensor): Target normal vectors of shape (batch_size, 3)
            
        #     Returns:
        #     torch.Tensor: Rotation matrices of shape (batch_size, 3, 3)
        #     """
        #     batch_size = source_norm.shape[0]
        #     device = source_norm.device

        #     # Ensure input vectors are normalized
        #     source_norm = F.normalize(source_norm, dim=1)
        #     target_norm = F.normalize(target_norm, dim=1)

        #     # Compute the dot product
        #     dot_product = torch.sum(source_norm * target_norm, dim=1)

        #     # If dot product is negative, flip the target vector
        #     flip_mask = dot_product < 0
        #     target_norm = torch.where(flip_mask.unsqueeze(1), -target_norm, target_norm)

        #     # Recompute dot product after potential flipping
        #     dot_product = torch.sum(source_norm * target_norm, dim=1)

        #     # Compute the axis of rotation (cross product)
        #     axis = torch.cross(source_norm, target_norm, dim=1)
        #     axis_norm = torch.norm(axis, dim=1, keepdim=True)

        #     # Clamp dot_product to [-1, 1] to avoid numerical issues
        #     dot_product = torch.clamp(dot_product, -1.0, 1.0)

        #     # Compute the angle
        #     angle = torch.acos(dot_product)

        #     # If the angle is very small, return identity matrix
        #     identity = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        #     small_angle_mask = axis_norm.squeeze(1) < 1e-6

        #     # Handle cases where source and target are opposite
        #     opposite_mask = (1.0 - dot_product) < 1e-6
        #     if opposite_mask.any():
        #         # Find an arbitrary perpendicular vector for rotation axis
        #         arbitrary_vec = torch.ones_like(source_norm)
        #         arbitrary_vec[:, 2] = -(source_norm[:, 0] + source_norm[:, 1]) / source_norm[:, 2]
        #         arbitrary_vec = F.normalize(arbitrary_vec, dim=1)
        #         axis = torch.where(opposite_mask.unsqueeze(1), arbitrary_vec, axis)

        #     # Normalize the axis
        #     axis = F.normalize(axis, dim=1)

        #     # Compute rotation matrix using Rodrigues' rotation formula
        #     k_times_angle = axis * angle.unsqueeze(1)
        #     k_cross = torch.zeros(batch_size, 3, 3, device=device)
        #     k_cross[:, 0, 1], k_cross[:, 0, 2] = -axis[:, 2], axis[:, 1]
        #     k_cross[:, 1, 0], k_cross[:, 1, 2] = axis[:, 2], -axis[:, 0]
        #     k_cross[:, 2, 0], k_cross[:, 2, 1] = -axis[:, 1], axis[:, 0]

        #     rotation_matrix = (
        #         identity +
        #         torch.sin(angle).unsqueeze(1).unsqueeze(2) * k_cross +
        #         (1 - torch.cos(angle)).unsqueeze(1).unsqueeze(2) * torch.bmm(k_cross, k_cross)
        #     )

        #     # Use identity matrix for small angles
        #     rotation_matrix = torch.where(small_angle_mask.unsqueeze(1).unsqueeze(2), identity, rotation_matrix)

        #     return rotation_matrix

        # def gauss_newton_optimization(L, verts):
        #     # use inexact Gauss-Newton method to update the vertices
        #     L = L.to_sparse_csr()
        #     delta = L.mm(rearrange(verts, 'b n c -> (b n) c'))
        #     LTL = torch.matmul(L.to_dense().t(), L.to_dense())
        #     LTL.diagonal().add_(1e-6)
        #     U, S, Vt = torch.linalg.svd(LTL.to(torch.float32))
        #     S_inv = torch.where(S > 1e-10, 1.0 / S, torch.zeros_like(S))
        #     verts = torch.matmul(Vt.t(), torch.matmul(U.t(), torch.matmul(L.to_dense().t(), delta)) * S_inv.unsqueeze(1)).to(torch.float32)
        #     verts = rearrange(verts, '(b n) c -> b n c', b=b)

        #     return verts

        template_mesh = load(self.super_params.template_mesh_dir)
        template_mesh = Meshes(
            verts=[torch.tensor(template_mesh.vertices, dtype=torch.float32)], 
            faces=[torch.tensor(template_mesh.faces, dtype=torch.int64)]
            ).to(DEVICE).extend(b)
        
        # sample and apply offset in two-stage manner
        # stage 1: smooth global offset
        verts = template_mesh.verts_padded()
        # find the rotation matrix that makes the centroid vector are in the same direction
        df_c = torch.stack([2 * (torch.nonzero(df <= 1).to(torch.float32).mean(0) / d - 0.5) 
                            for df in df_preds[:, 2]])[:, [1, 0, 2]]   # reorder dimensions
        mesh_c = self.mesh_c[1].unsqueeze(0).expand(b, -1)
        R = find_rotation_matrix_xz(mesh_c, df_c)
        verts = torch.bmm(R, verts.transpose(1, 2)).transpose(1, 2).to(torch.float32)

        template_mesh = template_mesh.update_padded(verts)

        # verts = template_mesh.verts_padded()
        # # align the tricuspid valve & pulmonary valve centroid & normal
        # df_rv_c = ((df_preds[:, 3] == 1) | (df_preds[:, 2] == 1)) ^ (df_preds[:, 2] == 1)
        # df_rv_c = torch.cat([RemoveSmallObjects(min_size=8)(i.unsqueeze(0)) for i in df_rv_c], dim=0)
        # # convert binary masks to point clouds
        # df_rv_c = [2 * (torch.nonzero(df).to(torch.float32) / d - 0.5) for df in df_rv_c]
        # df_rv_c = torch.nn.utils.rnn.pad_sequence(df_rv_c, batch_first=True)
        # df_rv_c = df_rv_c[:, :, [1, 0, 2]]  # Reorder dimensions
        # # find cluster centers
        # cluster_centers, cluster_normals, _ = find_cluster_centers_and_normals_batch(df_rv_c, max_clusters=4)
        # # find the tricuspid valve and pulmonary valve centroid
        # mesh_tv = verts[:, self.vert_label == 5]
        # mesh_tv_c = mesh_tv.mean(1)
        # mesh_pv = verts[:, self.vert_label == 6]
        # mesh_pv_c = mesh_pv.mean(1)
        # # find closest cluster centers to mesh centroids
        # idx_tv_c = find_closest_points(cluster_centers, mesh_tv_c)
        # idx_pv_c = find_closest_points(cluster_centers, mesh_pv_c)
        # df_tv_c = cluster_centers[torch.arange(cluster_centers.size(0)), idx_tv_c]
        # df_tv_norm = cluster_normals[torch.arange(cluster_normals.size(0)), idx_tv_c]
        # df_pv_c = cluster_centers[torch.arange(cluster_centers.size(0)), idx_pv_c]
        # df_pv_norm = cluster_normals[torch.arange(cluster_normals.size(0)), idx_pv_c]
        # # adjust vertices
        # mesh_tv_norm = find_cluster_normal(mesh_tv, mesh_tv_c)
        # R_tv = find_rotation_matrix_rodrigues(mesh_tv_norm, df_tv_norm)
        # verts[:, self.vert_label == 5] = torch.bmm(R_tv, (mesh_tv - mesh_tv_c.unsqueeze(1)).transpose(1, 2)).transpose(1, 2) + df_tv_c.unsqueeze(1)
        # mesh_pv_norm = find_cluster_normal(mesh_pv, mesh_pv_c)
        # R_pv = find_rotation_matrix_rodrigues(mesh_pv_norm, df_pv_norm)
        # verts[:, self.vert_label == 6] = torch.bmm(R_pv, (mesh_pv - mesh_pv_c.unsqueeze(1)).transpose(1, 2)).transpose(1, 2) + df_pv_c.unsqueeze(1)

        # verts = gauss_newton_optimization(template_mesh.laplacian_packed(), verts)
        # template_mesh = template_mesh.update_padded(verts)

        # stage 2: local offset
        verts = template_mesh.verts_padded()
        for i, l in zip([1, 0, 2, 0], [[0], [2], [1], [3]]): # lv-epi, lv, rv, rv-epi
            df_pred = df_preds[:, i]
            verts_idx = torch.any(torch.stack([self.vert_label == i for i in l]), dim=0)

            # calculate the gradient of the df
            direction = torch.gradient(-df_pred, dim=(1, 2, 3), edge_order=1)
            direction = torch.stack(direction, dim=1)
            direction /= (torch.norm(direction, dim=1, keepdim=True) + 1e-16)
            direction[torch.isnan(direction)] = 0
            direction[torch.isinf(direction)] = 0
            
            for _ in range(self.super_params.iteration):
                offset = F.grid_sample(
                    direction.permute(0, 1, 4, 2, 3), 
                    verts[:, verts_idx].unsqueeze(1).unsqueeze(1),
                    align_corners=False, padding_mode="zeros"
                ).view(b, 3, -1).transpose(-1, -2)[..., [1, 0, 2]]

                # transform from NDC space to pixel space
                verts = d * (verts / 2 + 0.5)
                verts[:, verts_idx] += offset
                # transform verts back to NDC space
                verts = 2 * (verts / d - 0.5)

        template_mesh = template_mesh.update_padded(verts)

        return template_mesh


    def load_pretrained_weight(self, phase):
        if phase == "unet" or phase == "all":
            encoder_mr_ckpt = torch.load(glob.glob(f"{self.super_params.use_ckpt}/trained_weights/best_UNet_MR.pth")[0], 
                                        map_location=DEVICE)
            encoder_ct_ckpt = torch.load(glob.glob(f"{self.super_params.use_ckpt}/trained_weights/best_UNet_CT.pth")[0], 
                                        map_location=DEVICE)
            self.encoder_mr.load_state_dict(encoder_mr_ckpt)
            self.encoder_ct.load_state_dict(encoder_ct_ckpt)
            print("Pretrained UNet loaded.")

        if phase == "resnet" or phase == "all":
            decoder_ckpt = torch.load(glob.glob(f"{self.super_params.use_ckpt}/trained_weights/best_ResNet.pth")[0], 
                                    map_location=DEVICE)
            self.decoder.load_state_dict(decoder_ckpt)
            print("Pretrained ResNet loaded.")

        if phase == "gsn" or phase == "all":
            GSN_ckpt = torch.load(glob.glob(f"{self.super_params.use_ckpt}/trained_weights/best_GSN.pth")[0], 
                                map_location=DEVICE)
            self.GSN.load_state_dict(GSN_ckpt)
            print("Pretrained GSN loaded.")


    def train_iter(self, epoch, phase):
        if phase == "unet":
            self.encoder_mr.train()
            self.encoder_ct.train()

            train_loss_epoch = dict(total=0.0, ct=0.0, mr=0.0)
            # train the CT segmentation encoder
            for step, data_ct in enumerate(self.ct_train_loader):
                img_ct, seg_true_ct = (
                    data_ct["ct_image"].as_tensor().to(DEVICE),
                    data_ct["ct_label"].as_tensor().to(DEVICE),
                    )

                self.optimzer_ct_unet.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    seg_pred_ct = sliding_window_inference(
                        img_ct, 
                        roi_size=self.super_params.crop_window_size, 
                        sw_batch_size=8, 
                        predictor=self.encoder_ct,
                        overlap=0.5, 
                        mode="gaussian", 
                    ) 
                    loss = self.dice_loss_fn(seg_pred_ct, seg_true_ct)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimzer_ct_unet)
                self.scaler.update()
                
                train_loss_epoch["ct"] += loss.item()
            
            train_loss_epoch["ct"] = train_loss_epoch["ct"] / (step + 1)
                
            self.lr_scheduler_ct_unet.step(train_loss_epoch["ct"])

            # train the CMR segmentation encoder
            for step, data_mr in enumerate(self.mr_train_loader):
                img_mr, seg_true_mr = (
                    data_mr["mr_image"].as_tensor().to(DEVICE),
                    data_mr["mr_label"].as_tensor().to(DEVICE),
                    )

                self.optimzer_mr_unet.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    seg_pred_mr = sliding_window_inference(
                        img_mr,
                        roi_size=self.super_params.crop_window_size[:2],
                        sw_batch_size=8,
                        predictor=self.encoder_mr,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    loss = self.dice_loss_fn(seg_pred_mr, seg_true_mr)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimzer_mr_unet)
                self.scaler.update()
                
                train_loss_epoch["mr"] += loss.item()

            train_loss_epoch["mr"] = train_loss_epoch["mr"] / (step + 1)
            self.lr_scheduler_mr_unet.step(train_loss_epoch["mr"])

            train_loss_epoch["total"] = train_loss_epoch["ct"] + train_loss_epoch["mr"]
            train_loss_epoch["seg"] = train_loss_epoch["total"]

            for k, v in self.unet_loss.items():
                self.unet_loss[k] = np.append(self.unet_loss[k], train_loss_epoch[k])

            wandb.log(
                {f"{phase}_loss": train_loss_epoch["total"]},
                step=epoch + 1
            )
            
        elif phase == "resnet":
            self.encoder_ct.eval()
            self.decoder.train()

            train_loss_epoch = dict(total=0.0, df=0.0)
            for step, data_ct in enumerate(self.ct_train_loader):
                img_ct, seg_true_ct = (
                    data_ct["ct_image"].to(DEVICE),
                    data_ct["ct_label"].to(DEVICE),
                )

                self.optimizer_resnet.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    seg_pred_ct = sliding_window_inference(
                        img_ct,
                        roi_size=self.super_params.crop_window_size,
                        sw_batch_size=8,
                        predictor=self.encoder_ct,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    seg_data = [self.post_transform({"pred": i, "label": j, "modal": "ct"}) for i, j in zip(seg_pred_ct, seg_true_ct)]
                    seg_pred_ct = torch.stack([i["pred"] for i in seg_data], dim=0)
                    seg_pred_ct_ds = F.interpolate(seg_pred_ct.as_tensor(), 
                                                    scale_factor=1 / self.super_params.pixdim[-1], 
                                                    mode="trilinear")
                    seg_true_ct = torch.stack([i["label"] for i in seg_data], dim=0)
                    seg_true_ct_ds = F.interpolate(seg_true_ct.as_tensor(),
                                                   scale_factor=1 / self.super_params.pixdim[-1],
                                                   mode="nearest-exact")
                    mask = (torch.argmax(seg_pred_ct_ds, dim=1, keepdim=True) == 0).detach()
                    seg_pred_ct_ds = ~mask * seg_pred_ct_ds + mask * self.decoder(seg_pred_ct_ds)

                    loss = self.msk_dice_loss_fn(seg_pred_ct_ds, seg_true_ct_ds, mask)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_resnet)
                self.scaler.update()
                
                train_loss_epoch["total"] += loss.item()
                train_loss_epoch["df"] += loss.item()

            for k, v in train_loss_epoch.items():
                train_loss_epoch[k] = v / (step + 1)
                self.resnet_loss[k] = np.append(self.resnet_loss[k], train_loss_epoch[k])

            wandb.log(
                {f"{phase}_loss": train_loss_epoch["total"]},
                step=epoch + 1
                )

            self.lr_scheduler_resnet.step(train_loss_epoch["total"])

        elif phase == "gsn":
            self.encoder_ct.eval()
            self.decoder.eval()
            self.GSN.train()

            finetune_loss_epoch = dict(total=0.0, chmf=0.0, smooth=0.0)
            for step, data_ct in enumerate(self.ct_train_loader):
                img_ct, seg_true_ct = (
                    data_ct["ct_image"].to(DEVICE),
                    data_ct["ct_label"].to(DEVICE)
                )
                seg_true_ct_ = torch.stack([self.post_transform({"label": i, "modal": "ct"})["label"] for i in seg_true_ct], dim=0)
                mesh_true_ct = self.surface_extractor(seg_true_ct_)

                self.optimizer_gsn.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    seg_pred_ct = sliding_window_inference(
                        img_ct,
                        roi_size=self.super_params.crop_window_size,
                        sw_batch_size=8,
                        predictor=self.encoder_ct,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    seg_pred_ct = torch.stack([self.post_transform({"pred": i, "label": j, "modal": "ct"})["pred"] 
                                               for i, j in zip(seg_pred_ct, seg_true_ct)], dim=0)
                    seg_pred_ct_ds = F.interpolate(seg_pred_ct.as_tensor(), 
                                                    scale_factor=1 / self.super_params.pixdim[-1], 
                                                    mode="trilinear")
                    mask = (torch.argmax(seg_pred_ct_ds, dim=1, keepdim=True) == 0).detach()
                    seg_pred_ct_ds = ~mask * seg_pred_ct_ds + mask * self.decoder(seg_pred_ct_ds)
                    seg_pred_ct_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ct_ds])
                    foreground = (seg_pred_ct_ds > 0)
                    lv = (seg_pred_ct_ds == 1)
                    rv = (seg_pred_ct_ds == 3)
                    myo = (seg_pred_ct_ds == 2)
                    df_pred_ct = torch.stack([
                        distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                        for i in [foreground, lv, rv, myo]], dim=1)
                    
                    template_mesh = self.warp_template_mesh(df_pred_ct.detach())
                    level_outs = self.GSN(template_mesh, self.subdivided_faces.faces_levels)

                    loss_chmf, loss_smooth = 0.0, 0.0
                    for l, subdiv_mesh in enumerate(level_outs):
                        verts_label = self.subdivided_faces.labels_levels[l]
                        for msh_idx, subdiv_idx in enumerate([[0], [1], [2, 3]]):  # lv, rv, myo
                            loss_chmf += chamfer_distance(
                                subdiv_mesh.verts_padded()[:, torch.any(torch.stack([verts_label == i for i in subdiv_idx]), dim=0)], 
                                mesh_true_ct[msh_idx].verts_padded(),
                                point_reduction="mean", batch_reduction="mean"
                                )[0] 
                        loss_smooth += mesh_laplacian_smoothing(subdiv_mesh, method="cotcurv")
                    
                    loss = self.super_params.lambda_0 * loss_chmf +\
                        self.super_params.lambda_1 * loss_smooth

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_gsn)
                self.scaler.update()
                
                finetune_loss_epoch["total"] += loss.item()
                finetune_loss_epoch["chmf"] += loss_chmf.item()
                finetune_loss_epoch["smooth"] += loss_smooth.item()

            for k, v in finetune_loss_epoch.items():
                finetune_loss_epoch[k] = v / (step + 1)
                self.gsn_loss[k] = np.append(self.gsn_loss[k], finetune_loss_epoch[k])

            wandb.log(
                {f"{phase}_loss": finetune_loss_epoch["total"]},
                step=epoch + 1
            )

            self.lr_scheduler_gsn.step(finetune_loss_epoch["total"])

        elif phase == "ndf":
            self.encoder_mr.eval()
            self.decoder.eval()
            self.GSN.eval()
            self.NDF.train()

            train_loss_epoch = dict(total=0.0, ndf=0.0)
            for step, data_mr in enumerate(self.mr_train_loader):
                img_mr, seg_true_mr = (
                    data_mr["mr_image"].to(DEVICE),
                    data_mr["mr_label"].to(DEVICE),
                )
                batch = data_mr["mr_batch"].item()
                bbox = generate_spatial_bounding_box(seg_true_mr)
                h, w = img_mr.shape[-2:]
                img_mr = img_mr[..., :, bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]
                seg_true_mr = seg_true_mr.unflatten(0, (batch, -1)).swapaxes(1, 2)

                try:
                    self.optimizer_ndf.zero_grad()
                    with torch.autocast(device_type=DEVICE):
                        seg_pred_mr = sliding_window_inference(
                            img_mr,
                            roi_size=self.super_params.crop_window_size[:2],
                            sw_batch_size=8,
                            predictor=self.encoder_mr,
                            overlap=0.5,
                            mode="gaussian",
                        )
                        seg_pred_mr = F.pad(seg_pred_mr, (bbox[0][2]-1, w-bbox[1][2]+1, bbox[0][1]-1, h-bbox[1][1]+1), "constant", 0)
                        seg_pred_mr = seg_pred_mr.unflatten(0, (batch, -1)).swapaxes(1, 2)
                        seg_pred_mr = torch.stack([self.post_transform({"pred": i, "label": j, "modal": "mr"})["pred"] 
                                                for i, j in zip(seg_pred_mr, seg_true_mr)], dim=0)
                        seg_pred_mr_ds = F.interpolate(seg_pred_mr.as_tensor(), 
                                                        scale_factor=1 / self.super_params.pixdim[-1], 
                                                        mode="trilinear")
                        mask = (torch.argmax(seg_pred_mr_ds, dim=1, keepdim=True) == 0)
                        seg_pred_mr_ds = ~mask * seg_pred_mr_ds + mask * self.decoder(seg_pred_mr_ds)
                        seg_pred_mr_ds = torch.stack([self.pred_transform(i) for i in seg_pred_mr_ds])
                        foreground = (seg_pred_mr_ds > 0)
                        lv = (seg_pred_mr_ds == 1)
                        rv = (seg_pred_mr_ds == 3)
                        myo = (seg_pred_mr_ds == 2)
                        df_pred_mr = torch.stack([
                            distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                            for i in [foreground, lv, rv, myo]], dim=1)
                        
                        template_mesh = self.warp_template_mesh(df_pred_mr.detach())
                        del df_pred_mr, seg_pred_mr, seg_pred_mr_ds, seg_true_mr, mask, foreground, myo, img_mr # release memory
                        torch.cuda.empty_cache()
                        try:
                            # method 1: NDF applied right after warping the control mesh
                            ndf_verts = self.NDF(template_mesh.verts_padded()[0], end_time=1, step=batch-1, invert=False)
                            loss_ndf = self.l1_loss_fn(ndf_verts, template_mesh.verts_padded())
                        except AssertionError:
                            loss_ndf = torch.nan

                        # template_mesh = template_mesh.update_padded(ndf_verts).detach()
                        # subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels)[-1]
                        # # method 2: NDF applied after the GSN
                        # ndf_verts = self.NDF(subdiv_mesh.verts_padded()[0], end_time=1, step=batch-1, invert=False)
                        # loss_ndf = self.l1_loss_fn(ndf_verts, subdiv_mesh.verts_padded())
                        # subdiv_mesh = subdiv_mesh.update_padded(ndf_verts)
                        
                        loss = loss_ndf

                        if loss != loss: continue

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer_ndf)
                    self.scaler.update()

                    train_loss_epoch["total"] += loss.item()
                    train_loss_epoch["ndf"] += loss_ndf.item()
                except RuntimeError:
                    id = os.path.basename(self.mr_train_loader.dataset.data[step]["mr_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')
                    print("Out of memory at", id, "| shape is ", data_mr["mr_label"].shape)
                    exit()

            for k, v in train_loss_epoch.items():
                train_loss_epoch[k] = v / (step + 1)
                self.ndf_loss[k] = np.append(self.ndf_loss[k], train_loss_epoch[k])

            wandb.log(
                {f"{phase}_loss": train_loss_epoch["total"]},
                step=epoch + 1
            )

            self.lr_scheduler_ndf.step(train_loss_epoch["total"])


    def valid(self, epoch, save_on):
        self.decoder.eval()
        self.GSN.eval()
        if self.super_params._4d:
            self.NDF.eval()
        
        # save model
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        torch.save(self.encoder_ct.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_UNet_CT.pth")
        torch.save(self.encoder_mr.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_UNet_MR.pth")
        torch.save(self.decoder.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_ResNet.pth")
        torch.save(self.GSN.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_GSN.pth")
        if self.super_params._4d:
            torch.save(self.NDF.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_NDF.pth")
        # save the subdivided_faces.faces_levels as pth file
        for level, faces in enumerate(self.subdivided_faces.faces_levels):
            torch.save(faces, f"{ckpt_weight_path}/{epoch+1}_subdivided_faces_l{level}.pth")
        
        # choose the validation loader
        if save_on == "sct":
            modal = "ct"
            encoder = self.encoder_ct
            valid_loader = self.ct_valid_loader
            roi_size = self.super_params.crop_window_size
        elif save_on == "cap":
            modal = "mr"
            encoder = self.encoder_mr
            valid_loader = self.mr_valid_loader
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError("Invalid dataset name")
        encoder.eval()

        df_metric_batch_decoder = MSEMetric(reduction="mean_batch")
        msh_metric_batch_decoder = DiceMetric(include_background=False, reduction="mean_batch")

        cached_data = dict()
        choice_case = np.random.choice(len(valid_loader), 1)[0]
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                img, seg_true, df_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                    data[f"{modal}_df"].as_tensor().to(DEVICE),
                )
                batch = data[f"{modal}_batch"].item() if save_on == "cap" else 2
                seg_true = seg_true.unflatten(0, (batch, -1)).swapaxes(1, 2) if save_on == "cap" else seg_true

                # evaluate the error between predicted df and the true df
                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=roi_size, 
                    sw_batch_size=8, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian", 
                )
                seg_pred = seg_pred.unflatten(0, (batch, -1)).swapaxes(1, 2) if save_on == "cap" else seg_pred
                seg_data = [self.post_transform({"pred": i, "label": j, "modal": modal}) for i, j in zip(seg_pred, seg_true)]
                seg_pred = torch.stack([i["pred"] for i in seg_data], dim=0)
                seg_pred_ds = F.interpolate(seg_pred.as_tensor(), 
                                                scale_factor=1 / self.super_params.pixdim[-1], 
                                                mode="trilinear")
                mask = (torch.argmax(seg_pred_ds, dim=1, keepdim=True) == 0).detach()
                seg_pred_ds = ~mask * seg_pred_ds + mask * self.decoder(seg_pred_ds)
                seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])
                foreground = (seg_pred_ds > 0)
                lv = (seg_pred_ds == 1)
                rv = (seg_pred_ds == 3)
                myo = (seg_pred_ds == 2)
                df_pred = torch.stack([
                    distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                    for i in [foreground, lv, rv, myo]], dim=1)
                
                df_metric_batch_decoder(df_pred, df_true)

                # evaluate the error between subdivided mesh and the true segmentation
                template_mesh = self.warp_template_mesh(df_pred)
                template_mesh_ = template_mesh.clone()

                if save_on == "cap" and self.super_params._4d:
                    # method 1: NDF applied right after warping the control mesh
                    ndf_verts = self.NDF(template_mesh.verts_padded()[0], end_time=1, step=batch-1, invert=False) 
                    template_mesh = template_mesh.update_padded(ndf_verts)

                subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels)[-1]
                
                # if save_on == "cap" and self.super_params._4d:
                #     # method 2: NDF applied after the GSN
                #     ndf_verts = self.NDF(subdiv_mesh.verts_padded()[0], end_time=1, step=batch-1, invert=False)
                #     subdiv_mesh = subdiv_mesh.update_padded(ndf_verts)
                
                voxeld_mesh = torch.cat([
                    self.rasterizer(
                        pred_mesh.verts_padded(), pred_mesh.faces_padded())
                    for pred_mesh in subdiv_mesh
                    ], dim=0)     
                seg_true = torch.stack([i["label"] for i in seg_data], dim=0)
                msh_metric_batch_decoder(voxeld_mesh, (seg_true == 2).to(torch.float32))

                if step == choice_case:
                    df_true = df_true
                    df_pred = df_pred
                    seg_pred = torch.stack([self.pred_transform(i) for i in seg_pred])

                    cached_data = {
                        "df_true": df_true[0].cpu(),
                        "df_pred": df_pred[0].cpu(),
                        "seg_pred": seg_pred[0].cpu(),
                        "seg_pred_ds": seg_pred_ds[0].cpu(),
                        "seg_true": seg_true[0].cpu(),
                        "seg_true_ds": F.interpolate(seg_true,
                                                    scale_factor=1 / self.super_params.pixdim[-1], 
                                                    mode="nearest-exact")[0].cpu(),
                        "subdiv_mesh": subdiv_mesh[0].cpu(),
                        "template_mesh": template_mesh_[0].cpu(),
                    }

        # log dice score
        self.eval_df_score["myo"] = np.append(self.eval_df_score["myo"], df_metric_batch_decoder.aggregate().cpu())
        self.eval_msh_score["myo"] = np.append(self.eval_msh_score["myo"], msh_metric_batch_decoder.aggregate().cpu())
        draw_train_loss(
            self.ndf_loss if self.super_params._4d else self.gsn_loss, 
            self.super_params, task_code="dynamic", phase="train"
            )
        draw_eval_score(self.eval_df_score, self.super_params, task_code="dynamic", module="df")
        draw_eval_score(self.eval_msh_score, self.super_params, task_code="dynamic", module="msh")
        wandb.log({
            "train_categorised_loss": wandb.Table(
            columns=[f"train_loss \u2193", f"eval_df_error \u2193", f"eval_msh_score \u2191"],
            data=[[
                wandb.Image(f"{self.super_params.ckpt_dir}/dynamic/{self.super_params.run_id}/train_loss.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/dynamic/{self.super_params.run_id}/eval_df_score.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/dynamic/{self.super_params.run_id}/eval_msh_score.png"),
                ]]
            )},
            step=epoch + 1
            )
        eval_score_epoch = msh_metric_batch_decoder.aggregate().mean()
        wandb.log({"eval_score": eval_score_epoch}, step=epoch + 1)


        if eval_score_epoch > self.best_eval_score:
            # save the best model
            torch.save(self.encoder_ct.state_dict(), f"{ckpt_weight_path}/best_UNet_CT.pth")
            torch.save(self.encoder_mr.state_dict(), f"{ckpt_weight_path}/best_UNet_MR.pth")
            torch.save(self.decoder.state_dict(), f"{ckpt_weight_path}/best_ResNet.pth")
            torch.save(self.GSN.state_dict(), f"{ckpt_weight_path}/best_GSN.pth")
            if self.super_params._4d:
                torch.save(self.NDF.state_dict(), f"{ckpt_weight_path}/best_NDF.pth")
            # save the subdivided_faces.faces_levels as pth file
            for level, faces in enumerate(self.subdivided_faces.faces_levels):
                torch.save(faces, f"{ckpt_weight_path}/best_subdivided_faces_l{level}.pth")
            self.best_eval_score = eval_score_epoch

            # save visualization when the eval score is the best
            wandb.log(
                {
                    "seg_true vs mesh_pred": wandb.Plotly(draw_plotly(
                        seg_true=cached_data["seg_true"], 
                        mesh_pred=cached_data["subdiv_mesh"]
                        )),
                    "seg_true vs seg_pred": wandb.Plotly(draw_plotly(
                        seg_true=cached_data["seg_true"], 
                        seg_pred=cached_data["seg_pred"]
                        )),
                    "seg_true_ds vs seg_pred_ds": wandb.Plotly(draw_plotly(
                        seg_true=cached_data["seg_true_ds"], 
                        seg_pred=cached_data["seg_pred_ds"]
                        )),
                    "template vs df_pred": wandb.Plotly(draw_plotly(
                        df_pred=cached_data["df_pred"],
                        mesh_pred=cached_data["template_mesh"],
                        mesh_c=self.mesh_c
                        )),
                    "seg_true_ds vs df_pred": wandb.Plotly(draw_plotly(
                        seg_true=cached_data["seg_true_ds"], 
                        df_pred=cached_data["df_pred"],
                        )),
                    "df true vs pred": wandb.Plotly(ff.create_distplot(
                        [cached_data["df_true"][-1].flatten().cpu().numpy(), 
                        cached_data["df_pred"][-1].flatten().cpu().numpy()],
                        group_labels=["df_true", "df_pred"],
                        colors=["#EF553B", "#3366CC"],
                        bin_size=0.1
                    )),
                },
                step=epoch + 1
            )
         

    @torch.no_grad()
    def test(self, save_on):
        # load networks
        self.encoder_ct.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_UNet_CT.pth")))
        self.encoder_mr.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_UNet_MR.pth")))
        self.decoder.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_ResNet.pth")))
        self.GSN.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_GSN.pth")))
        if self.super_params._4d:
            self.NDF.load_state_dict(
                torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_NDF.pth")))
        # load the subdivided_faces.faces_levels
        self.subdivided_faces.faces_levels = [torch.load(
            glob.glob(f"{self.ckpt_dir}/*_subdivided_faces_l{level}.pth")[-1]
            ) for level in range(self.super_params.subdiv_levels)]
        self.decoder.eval()
        self.GSN.eval()

        if save_on in "sct":
            modal = "ct"
            encoder = self.encoder_ct
            valid_loader = self.ct_test_loader
            roi_size = self.super_params.crop_window_size
        elif save_on == "cap":
            modal = "mr"
            encoder = self.encoder_mr
            valid_loader = self.mr_test_loader
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError("Invalid dataset name")
        encoder.eval()

        msh_metric_batch_decoder = DiceMetric(include_background=False, reduction="none")
        actual_heart_size_in_pixel = []

        choice_case = np.random.choice(len(valid_loader), 1)[0]
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                id = os.path.basename(valid_loader.dataset.data[step][f"{modal}_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')
                if save_on == "cap":
                    id = id.split('-')[0]

                img, seg_true, df_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                    data[f"{modal}_df"].as_tensor().to(DEVICE),
                )
                batch = data[f"{modal}_batch"].item() if save_on == "cap" else 2
                seg_true = seg_true.unflatten(0, (batch, -1)).swapaxes(1, 2) if save_on == "cap" else seg_true

                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=roi_size, 
                    sw_batch_size=1, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian", 
                )
                seg_pred = seg_pred.unflatten(0, (batch, -1)).swapaxes(1, 2) if save_on == "cap" else seg_pred
                seg_data = [self.post_transform({"pred": i, "label": j, "modal": modal}) for i, j in zip(seg_pred, seg_true)]
                seg_pred = torch.stack([i["pred"] for i in seg_data], dim=0)
                seg_pred_ds = F.interpolate(seg_pred.as_tensor(), 
                                                scale_factor=1 / self.super_params.pixdim[-1], 
                                                mode="trilinear")
                mask = (torch.argmax(seg_pred_ds, dim=1, keepdim=True) == 0).detach()
                seg_pred_ds = ~mask * seg_pred_ds + mask * self.decoder(seg_pred_ds)
                seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])
                foreground = (seg_pred_ds > 0)
                lv = (seg_pred_ds == 1)
                rv = (seg_pred_ds == 3)
                myo = (seg_pred_ds == 2)
                df_pred = torch.stack([
                    distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                    for i in [foreground, lv, rv, myo]], dim=1)

                template_mesh = self.warp_template_mesh(df_pred) 
                template_mesh_ = template_mesh.clone()

                if save_on == "cap" and self.super_params._4d:
                    # method 1: NDF applied right after warping the control mesh
                    ndf_verts = self.NDF(template_mesh.verts_padded()[0], end_time=1, step=batch-1, invert=False) 
                    template_mesh = template_mesh.update_padded(ndf_verts)

                subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels)[-1]
                
                voxeld_mesh = torch.cat([
                    self.rasterizer(
                        pred_mesh.verts_padded(), pred_mesh.faces_padded())
                    for pred_mesh in subdiv_mesh
                    ], dim=0)     
                seg_true = torch.stack([i["label"] for i in seg_data], dim=0)

                actual_heart_size_in_pixel.append(list(seg_true.applied_operations[3 if self.super_params.target == "acdc" else 4]["orig_size"]))

                seg_true = (seg_true == 2).to(torch.float32)
                msh_metric_batch_decoder(voxeld_mesh, seg_true)

                if subdiv_mesh._N > 2:
                    for i in range(subdiv_mesh._N):
                        # save each mesh as a time instance
                        save_obj(f"{self.out_dir}/{id} - {i:02d}.obj", 
                                 subdiv_mesh[i].verts_packed(), subdiv_mesh[i].faces_packed())
                elif subdiv_mesh._N == 2:
                    phases = ['ED', 'ES']
                    for i in range(subdiv_mesh._N):
                        # save each mesh as a time instance
                        save_obj(f"{self.out_dir}/{id}-{phases[i]}.obj", 
                                 subdiv_mesh[i].verts_packed(), subdiv_mesh[i].faces_packed())
                else:
                    save_obj(
                    f"{self.out_dir}/{id}.obj", 
                        subdiv_mesh.verts_packed(), subdiv_mesh.faces_packed()
                    )

                if step == choice_case:
                    seg_pred = torch.stack([self.pred_transform(i) for i in seg_pred])
                    seg_true_ds = F.interpolate(seg_true,
                                                scale_factor=1 / self.super_params.pixdim[-1], 
                                                mode="nearest-exact")[0].cpu()
                    # save visualization when the eval score is the best
                    wandb.log(
                        {
                            "seg_true vs mesh_pred": wandb.Plotly(draw_plotly(
                                seg_true=seg_true[0].cpu(), 
                                mesh_pred=subdiv_mesh[0].cpu()
                                )),
                            "seg_true vs seg_pred": wandb.Plotly(draw_plotly(
                                seg_true=seg_true[0].cpu(), 
                                seg_pred=seg_pred[0].cpu()
                                )),
                            "seg_true_ds vs seg_pred_ds": wandb.Plotly(draw_plotly(
                                seg_true=seg_true_ds, 
                                seg_pred=seg_pred_ds[0].cpu()
                                )),
                            "template vs df_pred": wandb.Plotly(draw_plotly(
                                df_pred=df_pred[0].cpu(),
                                mesh_pred=template_mesh_[0].cpu(),
                                mesh_c=self.mesh_c
                                )),
                            "seg_true_ds vs df_pred": wandb.Plotly(draw_plotly(
                                seg_true=seg_true_ds, 
                                df_pred=df_pred[0].cpu(),
                                )),
                            "df true vs pred": wandb.Plotly(ff.create_distplot(
                                [df_true[0, -1].flatten().cpu().numpy(), 
                                df_pred[0, -1].flatten().cpu().numpy()],
                                group_labels=["df_true", "df_pred"],
                                colors=["#EF553B", "#3366CC"],
                                bin_size=0.1
                            )),
                        }
                    )

        size_in_pixel = np.median(np.array(actual_heart_size_in_pixel), axis=0)
        wandb.log({"actual_heart_size h (pixel)": size_in_pixel[0]})
        wandb.log({"actual_heart_size w (pixel)": size_in_pixel[1]})
        wandb.log({"actual_heart_size d (pixel)": size_in_pixel[2]})
        wandb.log({"test_score": msh_metric_batch_decoder.aggregate().mean()})


    @torch.no_grad()
    def ablation_study(self, save_on):
        # load networks
        self.encoder_ct.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_UNet_CT.pth")))
        self.encoder_mr.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_UNet_MR.pth")))
        self.decoder.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_ResNet.pth")))
        self.GSN.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_GSN.pth")))
        # load the subdivided_faces.faces_levels
        self.subdivided_faces.faces_levels = [torch.load(
            glob.glob(f"{self.ckpt_dir}/*_subdivided_faces_l{level}.pth")[-1]
            ) for level in range(self.super_params.subdiv_levels)]
        self.decoder.eval()
        self.GSN.eval()

        if save_on in "sct":
            modal = "ct"
            encoder = self.encoder_ct
            valid_loader = self.ct_test_loader
            roi_size = self.super_params.crop_window_size
        elif save_on == "cap":
            modal = "mr"
            encoder = self.encoder_mr
            valid_loader = self.mr_test_loader
            roi_size = self.super_params.crop_window_size[:2]
        else:
            raise ValueError("Invalid dataset name")
        encoder.eval()

        os.makedirs(f"{self.out_dir}/{modal}", exist_ok=True)

        for i, data in enumerate(valid_loader):
            id = os.path.basename(valid_loader.dataset.data[i][f"{modal}_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')
            if save_on == "cap":
                id = id.split('-')[0]

            img, seg_true, df_true = (
                data[f"{modal}_image"].to(DEVICE),
                data[f"{modal}_label"].to(DEVICE),
                data[f"{modal}_df"].as_tensor().to(DEVICE),
            )
            batch = data[f"{modal}_batch"].item() if save_on == "cap" else 2
            seg_true = seg_true.unflatten(0, (batch, -1)).swapaxes(1, 2) if save_on == "cap" else seg_true
            seg_pred = sliding_window_inference(
                img, 
                roi_size=roi_size, 
                sw_batch_size=1, 
                predictor=encoder,
                overlap=0.5, 
                mode="gaussian", 
            )
            seg_pred = seg_pred.unflatten(0, (batch, -1)).swapaxes(1, 2) if save_on == "cap" else seg_pred
            seg_data = [self.post_transform({"pred": i, "label": j, "modal": modal}) for i, j in zip(seg_pred, seg_true)]
            seg_pred = torch.stack([i["pred"] for i in seg_data], dim=0)
            seg_pred_ds = F.interpolate(seg_pred.as_tensor(), 
                                            scale_factor=1 / self.super_params.pixdim[-1], 
                                            mode="trilinear")
            mask = (torch.argmax(seg_pred_ds, dim=1, keepdim=True) == 0).detach()
            seg_pred_ds = ~mask * seg_pred_ds + mask * self.decoder(seg_pred_ds)
            seg_pred_ds = torch.stack([self.pred_transform(i) for i in seg_pred_ds])
            seg_true = torch.stack([i["label"] for i in seg_data], dim=0)
            seg_true_ds = F.interpolate(seg_true,
                                        scale_factor=1 / self.super_params.pixdim[-1], 
                                        mode="nearest-exact")

            # ****** Distance Field Prediction ******
            foreground = (seg_pred_ds > 0)
            lv = (seg_pred_ds == 1)
            rv = (seg_pred_ds == 3)
            myo = (seg_pred_ds == 2)
            df_pred = torch.stack([
                distance_transform_edt(i[:, 0]) + distance_transform_edt(~i[:, 0]) 
                for i in [foreground, lv, rv, myo]], dim=1)

            if not self.super_params._4d:
                # Save the seg_pred_ds and seg_true_ds as nib file
                nib.save(nib.nifti1.Nifti1Image(seg_pred_ds[0, 0].cpu().numpy(), np.eye(4)), f"{self.out_dir}/{modal}/{id}_pred.nii.gz")
                nib.save(nib.nifti1.Nifti1Image(seg_true_ds[0, 0].cpu().numpy(), np.eye(4)), f"{self.out_dir}/{modal}/{id}_true.nii.gz")
            
                # save the prediction and true distance field as npy files
                np.save(f"{self.out_dir}/{modal}/{id}-df_true.npy", df_true[0].cpu().numpy())
                np.save(f"{self.out_dir}/{modal}/{id}-df_pred.npy", df_pred[0].cpu().numpy())

                if save_on == "sct":
                    # warped + adaptive
                    template_mesh = self.warp_template_mesh(df_pred)  
                    subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels)[-1]
                    save_obj(
                    f"{self.out_dir}/{modal}/{id}-adaptive.obj", 
                        subdiv_mesh.verts_packed(), subdiv_mesh.faces_packed()
                    )

                    # warped + Loop subdivided
                    template_mesh = self.warp_template_mesh(df_pred)
                    template_mesh = Trimesh(template_mesh.verts_packed().cpu().numpy(), template_mesh.faces_packed().cpu().numpy())
                    for _ in range(2): template_mesh = template_mesh.subdivide_loop()
                    save_obj(
                    f"{self.out_dir}/{modal}/{id}-loop.obj", 
                        torch.tensor(template_mesh.vertices), torch.tensor(template_mesh.faces)
                    )

                    # unwarp + Loop subdivided
                    template_mesh = self.template_mesh.to(DEVICE)
                    template_mesh = Trimesh(template_mesh.verts_packed().cpu().numpy(), template_mesh.faces_packed().cpu().numpy())
                    for _ in range(2): template_mesh = template_mesh.subdivide_loop()
                    save_obj(
                    f"{self.out_dir}/{modal}/{id}-unwarp_loop.obj", 
                        torch.tensor(template_mesh.vertices), torch.tensor(template_mesh.faces)
                    )

                    # unwarp (template mesh)
                    save_obj(
                        f"{self.out_dir}/{modal}/{id}-template_mesh.obj", 
                        self.template_mesh.verts_packed(), self.template_mesh.faces_packed()
                    )

            # ****** Increamental Subdivision from 0 --> 2 ******
            template_mesh = self.warp_template_mesh(df_pred)                             # level 0

            if not self.super_params._4d and save_on == "sct":
                save_obj(
                    f"{self.out_dir}/{modal}/{id}-level_0.obj", 
                    template_mesh.verts_packed(), template_mesh.faces_packed()
                )

            subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels)   # level 1 & 2: [Meshes, Meshes]

            if not self.super_params._4d and save_on == "sct":
                for level in range(2):
                    save_obj(
                        f"{self.out_dir}/{modal}/{id}-level_{level+1}.obj", 
                        subdiv_mesh[level].verts_packed(), subdiv_mesh[level].faces_packed()
                    )

            # ****** Compare outputs w/o NODE ******
            if self.super_params._4d:
                subdiv_mesh = subdiv_mesh[-1]
                for i in range(subdiv_mesh._N):
                    # save each mesh as a time instance
                    save_obj(f"{self.out_dir}/{modal}/{id} - {i:02d}.obj", 
                            subdiv_mesh[i].verts_packed(), subdiv_mesh[i].faces_packed())
            