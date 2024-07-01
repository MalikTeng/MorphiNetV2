import os, sys, json
from collections import OrderedDict
from itertools import chain
from trimesh import Trimesh
from trimesh import load_mesh
from trimesh.convex import convex_hull
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import save_obj
from pytorch3d.ops import sample_points_from_meshes, taubin_smoothing
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops.marching_cubes import marching_cubes
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MSEMetric
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, 
    AsDiscrete,
    KeepLargestConnectedComponent,
    RemoveSmallObjects,
    CropForeground,
    Resize,
    Spacing,
    SpatialPad,
    EnsureType, 
)
from monai.transforms.utils import distance_transform_edt, generate_spatial_bounding_box
from monai.utils import set_determinism
import wandb
import plotly.io as pio
import plotly.graph_objects as go
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
                {k: np.asarray([]) for k in ["total", "chmf", "norm", "smooth"]}
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
        transforms = [
            AsDiscrete(argmax=True),
            KeepLargestConnectedComponent(independent=True),
            RemoveSmallObjects(min_size=32),
            Spacing([1.5, -1, -1], mode="nearest"),
            CropForeground(),
            Resize(
                self.super_params.crop_window_size[0], 
                size_mode="longest", mode="nearest-exact"
                ),
            SpatialPad(
                self.super_params.crop_window_size[0], 
                method="end", mode="constant"
                ),
            EnsureType(),
        ]
        self.mr_post_transform = Compose(transforms)
        self.mr_label_transform = Compose(transforms[2:])
        _ = transforms.pop(2)
        self.ct_post_transform = Compose(transforms)

        self._data_warper(rotation=True)

        # import control mesh (NDC space, [-1, 1]) to compute the subdivision matrix
        control_mesh = load_mesh(super_params.control_mesh_dir)
        self.control_mesh = Meshes(
            verts=[torch.tensor(control_mesh.vertices, dtype=torch.float32)], 
            faces=[torch.tensor(control_mesh.faces, dtype=torch.int64)]
            ).to(DEVICE)
        # define the indices of df and segmentation so that loss is calculated based on matching label class
        if "lv" in super_params.control_mesh_dir.lower():
            self.df_indices = 0
            self.seg_indices = 1
        elif "myo" in super_params.control_mesh_dir.lower():
            self.df_indices = 1        # index of channel
            self.seg_indices = 2   # index of class
        elif "rv" in super_params.control_mesh_dir.lower():
            self.df_indices = 2
            self.seg_indices = 3
        else:
            raise ValueError("Invalid control_mesh_dir")

        self._prepare_modules()
        self._prepare_optimiser()

        self.rasterizer = Rasterize(self.super_params.crop_window_size) # tool for rasterizing mesh


    def _data_warper(self, rotation:bool):
        
        if self.is_training or self.super_params.save_on == "cap":
            print(f"Preparing MR data {'with' if rotation else 'without'} rotation...")
            with open(self.super_params.mr_json_dir, "r") as f:
                mr_train_transform, mr_valid_transform = self._prepare_transform(
                    ["mr_image", "mr_label"], "mr", rotation
                    )
                mr_train_ds, mr_valid_ds, mr_test_ds = self._prepare_dataset(
                    json.load(f), "mr", mr_train_transform, mr_valid_transform
                )
                self.mr_train_loader, self.mr_valid_loader, self.mr_test_loader = self._prepare_dataloader(
                    mr_train_ds, mr_valid_ds, mr_test_ds
                )

        if self.is_training or self.super_params.save_on == "sct":
            print(f"Preparing MR data {'with' if rotation else 'without'} rotation...")
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


    def _prepare_transform(self, keys, modal, rotation):
        train_transform = pre_transform(
            keys, modal, "train", rotation,
            self.super_params.crop_window_size,
            self.super_params.pixdim
            )
        valid_transform = pre_transform(
            keys, modal, "valid", rotation,
            self.super_params.crop_window_size,
            self.super_params.pixdim
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
                "ct_image": os.path.join(self.super_params.ct_data_dir, f"imagesTr", os.path.split(d["image"])[-1]),
                "ct_label": os.path.join(self.super_params.ct_data_dir, f"labelsTr", os.path.split(d["label"])[-1]),
            } for d in data_list]
        

    def _prepare_dataset(self, data_json, modal, train_transform, valid_transform):
        train_data = self._remap_abs_path(data_json["train_fold0"], modal, "Tr")[:10]
        valid_data = self._remap_abs_path(data_json["validation_fold0"], modal, "Tr")[:10]
        test_data = self._remap_abs_path(data_json["test"], modal, "Ts")[:10]

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
        self.encoder_mr = UNet(
            spatial_dims=2, in_channels=1, out_channels=self.super_params.num_classes + 1,  # include background
            channels=self.super_params.channels, strides=self.super_params.strides,
            kernel_size=3, up_kernel_size=3,
            num_res_units=0
        ).to(DEVICE)
        self.encoder_ct = UNet(
            spatial_dims=3, in_channels=1, out_channels=self.super_params.num_classes + 1,  # include background
            channels=self.super_params.channels, strides=self.super_params.strides,
            kernel_size=3, up_kernel_size=3,
            num_res_units=0
        ).to(DEVICE)
        self.decoder = ResNet(
            "basic", 
            self.super_params.layers, self.super_params.block_inplanes, 
            n_input_channels=1,
            conv1_t_size=3, conv1_t_stride=1, no_max_pool=True, shortcut_type="B",
            num_classes=1,
            feed_forward=False, 
        ).to(DEVICE)

        # create pre-computed subdivision matrix
        self.subdivided_faces = Subdivision(self.control_mesh, self.super_params.subdiv_levels)
        # initialise the subdiv module
        self.GSN = GSN(
            hidden_features=self.super_params.hidden_features_gsn, 
            num_layers=self.super_params.subdiv_levels if self.super_params.subdiv_levels > 0 else 2,
        ).to(DEVICE)


    def _prepare_optimiser(self):
        self.dice_loss_fn = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            )
        self.mse_loss_fn = nn.MSELoss()
        self.skeleton_loss_fn = skeleton_loss_fn

        # initialise the optimiser for unet
        self.optimzer_mr_unet = torch.optim.Adam(
            self.encoder_mr.parameters(), lr=self.super_params.lr
            )
        self.lr_scheduler_mr_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimzer_mr_unet, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
        self.optimzer_ct_unet = torch.optim.Adam(
            self.encoder_ct.parameters(), lr=self.super_params.lr
            )
        self.lr_scheduler_ct_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimzer_ct_unet, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for resnet
        self.optimizer_resnet = torch.optim.Adam(
            self.decoder.parameters(), 
            lr=self.super_params.lr
            )
        self.lr_scheduler_resnet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_resnet, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for gsn
        self.optimizer_gsn = torch.optim.Adam(
            self.GSN.parameters(), 
            lr=self.super_params.lr
            )
        self.lr_scheduler_gsn = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gsn, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )

        # initialise the gradient scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        torch.backends.cudnn.enabled = torch.backends.cudnn.is_available()
        torch.backends.cudnn.benchmark = torch.backends.cudnn.is_available()


    def surface_extractor(self, seg_true, use_skeleton: bool=False):
        """
            WARNING: this operation is non-differentiable.
            input:
                seg_true: ground truth segmentation.
            return:
                surface mesh with vertices and faces in NDC space [-1, 1].
        """
        if use_skeleton:
            seg_true = seg_true.squeeze(1).cpu().numpy().astype(np.float32)
            skeletons = [convex_hull(np.indices(seg.shape)[:, seg > 0].reshape(3, -1).T) for seg in seg_true]
            # transform from world space to NDC space and convert to torch tensor
            mesh_true = Meshes(
                verts=[torch.tensor(2 * (skeleton.vertices - seg_true.shape[-1] // 2) / seg_true.shape[-1], dtype=torch.float32) 
                       for skeleton in skeletons],
                faces=[torch.tensor(skeleton.faces, dtype=torch.int64) for skeleton in skeletons]
            ).to(DEVICE)
        else:
            verts, faces = marching_cubes(
                seg_true.squeeze(1).permute(0, 3, 1, 2), 
                isolevel=0.1,
                return_local_coords=True,
            )
            mesh_true = Meshes(verts, faces)
            mesh_true = taubin_smoothing(mesh_true, 0.77, -0.34, 30)

        return mesh_true


    @torch.no_grad()
    def warp_control_mesh(self, df_pred, t=1.36, one_step=False):
        """
            input:
                control_mesh: the control mesh in NDC space [-1, 1].
                df pred: the predicted df.
            return:
                warped control mesh with vertices and faces in NDC space.
        """
        df_pred = df_pred.squeeze(1)
        control_mesh = load_mesh(self.super_params.control_mesh_dir)
        control_mesh = Meshes(
            verts=[torch.tensor(control_mesh.vertices, dtype=torch.float32)], 
            faces=[torch.tensor(control_mesh.faces, dtype=torch.int64)]
            ).to(DEVICE).extend(df_pred.shape[0])

        # calculate the gradient of the df
        direction = torch.gradient(-df_pred, dim=(1, 2, 3), edge_order=1)
        direction = torch.stack(direction, dim=1)
        direction /= (torch.norm(direction, dim=1, keepdim=True) + 1e-16)
        direction[torch.isnan(direction)] = 0
        direction[torch.isinf(direction)] = 0

        # sample and apply offset in two-step manner: smooth global -> sharp local
        verts = control_mesh.verts_padded()
        for step in range(2) if not one_step else [0]:
            if step == 0:
                # sampled offset is very fuzzy and apply it directly to the control mesh will break its manifold, so try average the offset and apply it
                offset = torch.stack([2 * (torch.nonzero(df <= 1).to(torch.float32).mean(0) / df.shape[-1] - 0.5) for df in df_pred])[:, [1, 0, 2]] - verts.mean(1)   # i, j, k -> x, y, z
                offset = offset.unsqueeze(1).expand(-1, control_mesh._V, -1)
            else:
                # too large the offset that applies close to valve will break the manifold, so try to reduce the offset with equation y = x * e^(-|x|/t) where t is temperature term
                offset = direction * df_pred.unsqueeze(1)
                offset = F.grid_sample(
                    offset.permute(0, 1, 4, 2, 3), 
                    verts.unsqueeze(1).unsqueeze(1),
                    align_corners=True
                ).view(df_pred.shape[0], 3, -1).transpose(-1, -2)[..., [1, 0, 2]]
                offset = offset * torch.exp(-abs(offset) / t)
                # transform from NDC space to pixel space
                verts = (verts / 2 + 0.5) * df_pred.shape[-1]
            verts += offset

        # transform verts back to NDC space and update the control mesh
        verts = 2 * (verts / df_pred.shape[-1] - 0.5)
        control_mesh = control_mesh.update_padded(verts)

        return control_mesh


    @torch.no_grad()
    def update_precomputed_faces(self):
        self.GSN.eval()
        
        faces_score_epoch = 0.0
        for step, data_ct in enumerate(self.ct_train_loader):
            seg_true_ct, df_true_ct = (
                data_ct["ct_label"].as_tensor().to(DEVICE),
                data_ct["ct_df"].as_tensor().to(DEVICE),
            )
            seg_true_ct = (seg_true_ct == self.seg_indices).to(torch.float32)
            mesh_true_ct = self.surface_extractor(seg_true_ct)

            control_mesh = self.warp_control_mesh(df_true_ct)
            subdiv_mesh = self.GSN(control_mesh, self.subdivided_faces.faces_levels)

            true_points = sample_points_from_meshes(mesh_true_ct, 2 * subdiv_mesh._F)
            pointcloud_true_ct = Pointclouds(points=true_points)
            faces_score_epoch += face_score(meshes=subdiv_mesh, pcls=pointcloud_true_ct)
        
        faces_score_epoch = faces_score_epoch / (step + 1)
        faces_score_epoch = [None, faces_score_epoch > 0.5]   # all faces will bu subdivided for at least one level.
        self.subdivided_faces = Subdivision(mesh=self.control_mesh, num_layers=2, allow_subdiv_faces=faces_score_epoch)


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
                        sw_batch_size=1, 
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
                        sw_batch_size=1,
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
            self.lr_scheduler_ct_unet.step(train_loss_epoch["mr"])

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
                img_ct, df_true_ct = (
                    data_ct["ct_image"].as_tensor().to(DEVICE),
                    data_ct["ct_df"].as_tensor().to(DEVICE),
                )
                df_true_ct = df_true_ct[:, self.df_indices].unsqueeze(1)

                self.optimizer_resnet.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    seg_pred_ct = sliding_window_inference(
                        img_ct,
                        roi_size=self.super_params.crop_window_size,
                        sw_batch_size=1,
                        predictor=self.encoder_ct,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    # downsample so that the resolution is the same as CMR data
                    seg_pred_ct = torch.stack([self.ct_post_transform(i) for i in seg_pred_ct], dim=0)
                    seg_pred_ct = (seg_pred_ct == self.seg_indices).to(torch.float32)
                    seg_pred_ct_ds = F.interpolate(seg_pred_ct, 
                                                    scale_factor=1 / self.super_params.pixdim[-1], 
                                                    mode="nearest-exact")
                    # compute the distance field
                    df_pred_ct_ = distance_transform_edt(seg_pred_ct_ds.squeeze(1)) + distance_transform_edt(1 - seg_pred_ct_ds.squeeze(1))
                    # predict the df + residual
                    df_pred_ct = self.decoder(df_pred_ct_.unsqueeze(1))

                    loss = self.mse_loss_fn(df_pred_ct, df_true_ct)

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

            finetune_loss_epoch = dict(total=0.0, chmf=0.0, norm=0.0, smooth=0.0)
            for step, data_ct in enumerate(self.ct_train_loader):
                img_ct, seg_true_ct, df_true_ct = (
                    data_ct["ct_image"].as_tensor().to(DEVICE),
                    data_ct["ct_label"].as_tensor().to(DEVICE),
                    data_ct["ct_df"].as_tensor().to(DEVICE),
                )
                img_ct.applied_operations = []
                seg_true_ct = (seg_true_ct == self.seg_indices).to(torch.float32)
                mesh_true_ct = self.surface_extractor(seg_true_ct)
                df_true_ct = df_true_ct[:, self.df_indices].unsqueeze(1)

                self.optimizer_gsn.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    seg_pred_ct = sliding_window_inference(
                        img_ct,
                        roi_size=self.super_params.crop_window_size,
                        sw_batch_size=1,
                        predictor=self.encoder_ct,
                        overlap=0.5,
                        mode="gaussian",
                    )
                    seg_pred_ct = torch.stack([self.ct_post_transform(i) for i in seg_pred_ct], dim=0)
                    seg_pred_ct = (seg_pred_ct == self.seg_indices).to(torch.float32)
                    seg_pred_ct_ds = F.interpolate(seg_pred_ct, 
                                                    scale_factor=1 / self.super_params.pixdim[-1], 
                                                    mode="nearest-exact")
                    df_pred_ct_ = distance_transform_edt(seg_pred_ct_ds.squeeze(1)) + distance_transform_edt(1 - seg_pred_ct_ds.squeeze(1))
                    df_pred_ct = self.decoder(df_pred_ct_.unsqueeze(1))
                    
                    control_mesh = self.warp_control_mesh(df_pred_ct.detach())
                    subdiv_mesh = self.GSN(control_mesh, self.subdivided_faces.faces_levels)

                    loss_chmf, _ = chamfer_distance(
                        mesh_true_ct.verts_padded(), subdiv_mesh.verts_padded(),
                        point_reduction="mean", batch_reduction="mean")
                    true_points = sample_points_from_meshes(mesh_true_ct, 2 * subdiv_mesh._F)
                    pointcloud_true_ct = Pointclouds(points=true_points)
                    loss_norm = point_mesh_face_distance(subdiv_mesh, pointcloud_true_ct, min_triangle_area=1e-3)
                    loss_smooth = mesh_laplacian_smoothing(subdiv_mesh, method="cotcurv")
                    
                    loss = self.super_params.lambda_[0] * loss_chmf +\
                            self.super_params.lambda_[1] * loss_norm +\
                                self.super_params.lambda_[2] * loss_smooth

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_gsn)
                self.scaler.update()
                
                finetune_loss_epoch["total"] += loss.item()
                finetune_loss_epoch["chmf"] += loss_chmf.item()
                finetune_loss_epoch["norm"] += loss_norm.item()
                finetune_loss_epoch["smooth"] += loss_smooth.item()

            for k, v in finetune_loss_epoch.items():
                finetune_loss_epoch[k] = v / (step + 1)
                self.gsn_loss[k] = np.append(self.gsn_loss[k], finetune_loss_epoch[k])

            wandb.log(
                {f"{phase}_loss": finetune_loss_epoch["total"]},
                step=epoch + 1
            )

            self.lr_scheduler_gsn.step(finetune_loss_epoch["total"])


    def valid(self, epoch, save_on):
        self.decoder.eval()
        self.GSN.eval()
        
        # choose the validation loader
        if save_on == "sct":
            modal = "ct"
            encoder = self.encoder_ct
            encoder.load_state_dict(self.encoder_ct.state_dict())
            encoder.eval()
            valid_loader = self.ct_valid_loader
        elif save_on == "cap":
            modal = "mr"
            encoder = self.encoder_mr
            encoder.load_state_dict(self.encoder_mr.state_dict())
            encoder.eval()
            valid_loader = self.mr_valid_loader
        else:
            raise ValueError("Invalid dataset name")

        df_metric_batch_decoder = MSEMetric(reduction="mean_batch")
        msh_metric_batch_decoder = DiceMetric(reduction="mean_batch")

        cached_data = dict()
        choice_case = np.random.choice(len(valid_loader), 1)[0]
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                img, seg_true, df_true = (
                    data[f"{modal}_image"].to(DEVICE),
                    data[f"{modal}_label"].to(DEVICE),
                    data[f"{modal}_df"].as_tensor().to(DEVICE),
                )
                seg_true = seg_true.unflatten(0, (2, -1)).swapaxes(1, 2)
                seg_true = torch.stack([self.mr_label_transform(i) for i in seg_true], dim=0)
                seg_true = (seg_true == self.seg_indices).to(torch.float32)
                df_true = df_true[:, self.df_indices].unsqueeze(1)

                # evaluate the error between predicted df and the true df
                seg_pred = sliding_window_inference(
                    img, 
                    roi_size=self.super_params.crop_window_size[:2], 
                    sw_batch_size=1, 
                    predictor=encoder,
                    overlap=0.5, 
                    mode="gaussian", 
                )
                seg_pred = seg_pred.unflatten(0, (2, -1)).swapaxes(1, 2)
                seg_pred = torch.stack([self.mr_post_transform(i) for i in seg_pred], dim=0)
                seg_pred = (seg_pred == self.seg_indices).to(torch.float32)
                seg_pred_ds = F.interpolate(seg_pred, 
                                                scale_factor=1 / self.super_params.pixdim[-1], 
                                                mode="nearest-exact")
                df_pred_ = distance_transform_edt(seg_pred_ds.squeeze(1)) + distance_transform_edt(1 - seg_pred_ds.squeeze(1))
                df_pred = self.decoder(df_pred_.unsqueeze(1))
                df_metric_batch_decoder(df_pred, df_true)

                # evaluate the error between subdivided mesh and the true segmentation
                control_mesh = self.warp_control_mesh(df_pred)  
                subdiv_mesh = self.GSN(control_mesh, self.subdivided_faces.faces_levels)
                voxeld_mesh = torch.cat([
                    self.rasterizer(
                        pred_mesh.verts_padded(), pred_mesh.faces_padded())
                    for pred_mesh in subdiv_mesh
                    ], dim=0)     
                msh_metric_batch_decoder(voxeld_mesh, seg_true)

                if step == choice_case:
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
                    }

        # log dice score
        self.eval_df_score["myo"] = np.append(self.eval_df_score["myo"], df_metric_batch_decoder.aggregate().cpu())
        self.eval_msh_score["myo"] = np.append(self.eval_msh_score["myo"], msh_metric_batch_decoder.aggregate().cpu())
        draw_train_loss(self.gsn_loss, self.super_params, task_code="dynamic", phase="gsn")
        draw_eval_score(self.eval_df_score, self.super_params, task_code="dynamic", module="df")
        draw_eval_score(self.eval_msh_score, self.super_params, task_code="dynamic", module="msh")
        wandb.log({
            "train_categorised_loss": wandb.Table(
            columns=[f"gsn_loss \u2193", f"eval_df_error \u2193", f"eval_msh_score \u2191"],
            data=[[
                wandb.Image(f"{self.super_params.ckpt_dir}/dynamic/{self.super_params.run_id}/gsn_loss.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/dynamic/{self.super_params.run_id}/eval_df_score.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/dynamic/{self.super_params.run_id}/eval_msh_score.png"),
                ]]
            )},
            step=epoch + 1
            )
        eval_score_epoch = msh_metric_batch_decoder.aggregate().mean()
        wandb.log({"eval_score": eval_score_epoch}, step=epoch + 1)

        # save model
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        torch.save(self.encoder_ct.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_UNet_CT.pth")
        torch.save(self.encoder_mr.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_UNet_MR.pth")
        torch.save(self.decoder.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_ResNet.pth")
        torch.save(self.GSN.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_GSN.pth")
        # save the subdivided_faces.faces_levels as pth file
        for level, faces in enumerate(self.subdivided_faces.faces_levels):
            torch.save(faces, f"{ckpt_weight_path}/{epoch+1}_subdivided_faces_l{level}.pth")

        # save visualization when the eval score is the best
        if eval_score_epoch > self.best_eval_score:
            self.best_eval_score = eval_score_epoch
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
                    "seg_true_ds vs df_pred": wandb.Plotly(draw_plotly(
                        seg_true=cached_data["seg_true_ds"], 
                        df_pred=cached_data["df_pred"],
                        mesh_pred=self.control_mesh.clone().cpu(),
                        )),
                    "df true vs pred": wandb.Plotly(ff.create_distplot(
                        [cached_data["df_true"][0].flatten().cpu().numpy(), 
                        cached_data["df_pred"][0].flatten().cpu().numpy()],
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
        self.AE.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_AutoEncoder.pth")))
        self.GSN.load_state_dict(
            torch.load(os.path.join(self.ckpt_dir, f"{self.super_params.best_epoch}_GSN.pth")))
        # load the subdivided_faces.faces_levels
        self.subdivided_faces.faces_levels = [torch.load(
            f"{self.ckpt_dir}/{self.super_params.best_epoch}_subdivided_faces_l{level}.pth"
            ) for level in range(self.super_params.subdiv_levels)]
        self.encoder_mr.eval()
        self.AE.eval()
        self.GSN.eval()

        if save_on in "sct":
            modal = "ct"
            self.AE.encoder.load_state_dict(self.encoder_ct.state_dict())
            valid_loader = self.ct_test_loader
        elif save_on == "cap":
            modal = "mr"
            self.AE.encoder.load_state_dict(self.encoder_mr.state_dict())
            valid_loader = self.mr_test_loader
        else:
            raise ValueError("Invalid dataset name")

        for i, data in enumerate(valid_loader):
            id = os.path.basename(valid_loader.dataset.data[i][f"{modal}_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')
            id = id.split('-')[0]
            img = data[f"{modal}_image"].as_tensor().to(DEVICE)

            df_pred, _ = self.AE(img)
            control_mesh = self.warp_control_mesh(df_pred)  
            subdiv_mesh = self.GSN(control_mesh, self.subdivided_faces.faces_levels)

            # smoothing the generated mesh
            subdiv_mesh = taubin_smoothing(subdiv_mesh, 0.77, -0.34, 30)
            
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

