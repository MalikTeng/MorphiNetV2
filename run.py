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
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MSEMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, 
    EnsureType, 
    AsDiscrete,
    KeepLargestConnectedComponent,
)
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
            self.ckpt_dir = os.path.join(super_params.ckpt_dir, "stationary", super_params.run_id)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.pretrain_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "seg"]}
            )
            self.train_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "df", "chmf", "norm", "smooth"]}
            )
            self.fine_tune_loss = {"hd": np.asarray([])}
            self.eval_df_score = OrderedDict(
                {k: np.asarray([]) for k in ["myo"]}
            )
            self.eval_msh_score = self.eval_df_score.copy()
            self.best_eval_score = 0
        else:
            self.ckpt_dir = super_params.ckpt_dir
            self.out_dir = super_params.out_dir
            os.makedirs(self.out_dir, exist_ok=True)

        self.post_transform = Compose([
            AsDiscrete(argmax=True),
            KeepLargestConnectedComponent(is_onehot=False),
            EnsureType(data_type="tensor", dtype=torch.float32, device=DEVICE)
            ])

        if not is_training and super_params.save_on != "cap":
            pass
        else:
            with open(super_params.mr_json_dir, "r") as f:
                mr_train_transform, mr_valid_transform = self._prepare_transform(["mr_image", "mr_label"], "mr")
                mr_train_ds, mr_valid_ds, mr_test_ds = self._prepare_dataset(
                    json.load(f), "mr", mr_train_transform, mr_valid_transform
                )
                self.mr_train_loader, self.mr_valid_loader, self.mr_test_loader = self._prepare_dataloader(
                    mr_train_ds, mr_valid_ds, mr_test_ds
                )
        
        if not is_training and super_params.save_on != "sct":
            pass
        else:
            with open(super_params.ct_json_dir, "r") as f:
                ct_train_transform, ct_valid_transform = self._prepare_transform(["ct_image", "ct_label"], "ct")
                ct_train_ds, ct_valid_ds, ct_test_ds = self._prepare_dataset(
                    json.load(f), "ct", ct_train_transform, ct_valid_transform
                )
                self.ct_train_loader, self.ct_valid_loader, self.ct_test_loader = self._prepare_dataloader(
                    ct_train_ds, ct_valid_ds, ct_test_ds
                )

        # import control mesh (NDC space, [-1, 1]) to compute the subdivision matrix
        control_mesh = load_mesh(super_params.control_mesh_dir)
        self.control_mesh = Meshes(
            verts=[torch.tensor(control_mesh.vertices, dtype=torch.float32)], 
            faces=[torch.tensor(control_mesh.faces, dtype=torch.int64)]
            )
        # define the indices of df and segmentation so that loss is calculated based on matching label class
        if "lv" in super_params.control_mesh_dir.lower():
            self.df_indices = 0
            self.seg_indices = 1
        elif "myo" in super_params.control_mesh_dir.lower():
            self.df_indices = 1        # index of channel
            self.seg_indices = "myo"   # index of class
        elif "rv" in super_params.control_mesh_dir.lower():
            self.df_indices = 2
            self.seg_indices = 3
        else:
            raise ValueError("Invalid control_mesh_dir")

        self._prepare_modules()
        self._prepare_optimiser()

        self.rasterizer = Rasterize(self.super_params.crop_window_size) # tool for rasterizing mesh


    def _prepare_transform(self, keys, modal):
        train_transform = pre_transform(
            keys, modal, "train",
            self.super_params.crop_window_size,
            self.super_params.pixdim, 
            )
        valid_transform = pre_transform(
            keys, modal, "valid",
            self.super_params.crop_window_size,
            self.super_params.pixdim, 
            )
        
        return train_transform, valid_transform


    def _remap_abs_path(self, data_list, modal):
        if modal == "mr":
            return [{
                "mr_image": os.path.join(self.super_params.mr_data_dir, "imagesTr", os.path.basename(d["image"])),
                "mr_label": os.path.join(self.super_params.mr_data_dir, "labelsTr", os.path.basename(d["label"])),
                "mr_slice_info": os.path.join(self.super_params.mr_data_dir, "slice_info", os.path.basename(d["label"])).replace(".nii.gz", "-info.npy"),
            } for d in data_list]
        elif modal == "ct":
            return [{
                "ct_image": os.path.join(self.super_params.ct_data_dir, "imagesTr", os.path.split(d["image"])[-1]),
                "ct_label": os.path.join(self.super_params.ct_data_dir, "labelsTr", os.path.split(d["label"])[-1]),
            } for d in data_list]
        

    def _prepare_dataset(self, data_json, modal, train_transform, valid_transform):
        train_data = self._remap_abs_path(data_json["train_fold0"], modal)
        valid_data = self._remap_abs_path(data_json["validation_fold0"], modal)
        test_data = self._remap_abs_path(data_json["test"], modal)

        train_ds = Dataset(
            train_data, train_transform, self.seed, sys.maxsize,
            self.super_params.cache_rate, self.num_workers
            )
        valid_ds = Dataset(
            valid_data, valid_transform, self.seed, sys.maxsize,
            self.super_params.cache_rate, self.num_workers
            )
        test_ds = Dataset(
            test_data, valid_transform, self.seed, sys.maxsize,
            self.super_params.cache_rate, self.num_workers
            )
        
        return train_ds, valid_ds, test_ds


    def _prepare_dataloader(self, train_ds, valid_ds, test_ds):
        if train_ds.__len__() > 0:
            train_loader = DataLoader(
                train_ds, batch_size=self.super_params.batch_size,
                shuffle=True, num_workers=self.num_workers,
                )
        else:
            train_loader = None
        if valid_ds.__len__() > 0:
            val_loader = DataLoader(
                valid_ds, batch_size=1,
                shuffle=False, num_workers=self.num_workers,
                )
        else:
            val_loader = None
        test_loader = DataLoader(
            test_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers,
            )
        
        return train_loader, val_loader, test_loader


    def _prepare_modules(self):
        # initialise the df-predict module
        self.encoder_mr = UNet(
            spatial_dims=3, in_channels=1, out_channels=self.super_params.num_classes + 1,  # include background
            channels=self.super_params.channels, 
            strides=self.super_params.strides,
            kernel_size=3, up_kernel_size=3,
            num_res_units=0
        ).to(DEVICE)
        self.encoder_ct = UNet(
            spatial_dims=3, in_channels=1, out_channels=self.super_params.num_classes + 1,  # include background
            channels=self.super_params.channels, 
            strides=self.super_params.strides,
            kernel_size=3, up_kernel_size=3,
            num_res_units=0
        ).to(DEVICE)
        self.AE = AutoEncoder(
            UNet(
                spatial_dims=3, in_channels=1, out_channels=self.super_params.num_classes + 1,  # include background
                channels=self.super_params.channels, 
                strides=self.super_params.strides,
                kernel_size=3, up_kernel_size=3,
                num_res_units=0
            ),
            ResNet(
                "basic", 
                self.super_params.layers, self.super_params.block_inplanes, 
                n_input_channels=self.super_params.num_classes + 1,
                conv1_t_size=3, conv1_t_stride=1, no_max_pool=True, shortcut_type="B",
                num_classes=self.super_params.num_classes - 1,
                feed_forward=False, 
            ),
            self.super_params.pixdim[0]
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

        # initialise the optimiser for ptretrain
        self.optimizer_pretrain = torch.optim.Adam(
            chain(self.encoder_mr.parameters(), self.encoder_ct.parameters()),
            lr=10 * self.super_params.lr
            )
        self.lr_scheduler_pretrain = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_pretrain, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for train
        self.optimizer_train = torch.optim.Adam(
            chain(self.AE.parameters(), self.GSN.parameters()), 
            lr=self.super_params.lr
            )
        self.lr_scheduler_train = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_train, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for fine-tune
        self.optimizer_fine_tune = torch.optim.Adam(
            chain(self.AE.parameters(), self.GSN.parameters()), 
            lr=self.super_params.lr
        )
        self.lr_scheduler_fine_tune = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fine_tune, mode="min", factor=0.1, patience=5, verbose=False,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
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
        seg_true = torch.logical_or(seg_true == 2, seg_true == 4) if isinstance(self.seg_indices, str) else (seg_true == self.seg_indices)
        # seg_true = (seg_true > 0) if isinstance(self.seg_indices, str) else (seg_true == self.seg_indices)
        if not use_skeleton:
            seg_true = seg_true.float()
            verts, faces = marching_cubes(
                seg_true.squeeze(1).permute(0, 3, 1, 2), 
                isolevel=0.1,
                return_local_coords=True,
            )
            # convert coordinates from (z, x, y) to (x, y, z)
            verts = [vert[:, [1, 0, 2]] for vert in verts]
            mesh_true = Meshes(verts, faces)
            mesh_true = taubin_smoothing(mesh_true, 0.77, -0.34, 30)
        else:
            seg_true = seg_true.float().squeeze(1).cpu().numpy()
            skeletons = [convex_hull(np.indices(seg.shape)[:, seg > 0].reshape(3, -1).T) for seg in seg_true]
            # transform from world space to NDC space and convert to torch tensor
            mesh_true = Meshes(
                verts=[torch.tensor(2 * (skeleton.vertices - skeleton.centroid) / skeleton.extents.max(), dtype=torch.float32) for skeleton in skeletons],
                faces=[torch.tensor(skeleton.faces, dtype=torch.int64) for skeleton in skeletons]
            ).to(DEVICE)

        return mesh_true


    @torch.no_grad()
    def warp_control_mesh(self, df_pred, t=2.73, one_step=False):
        """
            input:
                control_mesh: the control mesh in NDC space [-1, 1].
                df pred: the predicted df.
            return:
                warped control mesh with vertices and faces in NDC space.
        """
        df_pred = df_pred[:, self.df_indices].unsqueeze(1)
        control_mesh = self.control_mesh.extend(df_pred.shape[0]).to(DEVICE)

        # calculate the gradient of the df
        direction = torch.cat(torch.gradient(df_pred, dim=(2, 3, 4)), dim=1)
        direction /= (torch.norm(direction, dim=1, keepdim=True) + 1e-16) * -1

        # sample and apply offset in two-step manner: smooth global -> sharp local
        for step in range(2) if not one_step else [0]:
            coords = control_mesh.verts_padded()
            offset = direction * df_pred
            offset = F.grid_sample(
                offset, 
                torch.flip(coords, dims=(-1,)).unsqueeze(1).unsqueeze(1),
                align_corners=True
            ).view(df_pred.shape[0], 3, -1).permute(0, 2, 1)
            if step == 0:
                # sampled offset is very fuzzy and apply it directly to the control mesh will break its manifold, so try average the offset and apply it
                offset = offset.mean(dim=1, keepdim=True).expand(-1, control_mesh._V, -1)
            else:
                # too large the offset that applies close to valve will break the manifold, so try to reduce the offset with equation y = x * e^(-x/t) where t is temperature term
                offset *= torch.exp(-torch.sign(offset) * offset / t)
            # apply offset to the vertices
            control_mesh.scale_verts_(df_pred.shape[-1] / 2)
            control_mesh.offset_verts_(offset.reshape(-1, 3))
            # rescale back to NDC space
            control_mesh.scale_verts_(2 / df_pred.shape[-1])

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
        if phase == "pretrain":
            self.encoder_mr.train()
            self.encoder_ct.train()

            train_loss_epoch = dict(total=0.0, seg=0.0)
            for step, (data_ct, data_mr) in enumerate(zip(self.ct_train_loader, self.mr_train_loader)):
                img_ct, seg_true_ct = (
                    data_ct["ct_image"].as_tensor().to(DEVICE),
                    data_ct["ct_label"].as_tensor().to(DEVICE),
                    )
                img_mr, seg_true_mr = (
                    data_mr["mr_image"].as_tensor().to(DEVICE),
                    data_mr["mr_label"].as_tensor().to(DEVICE),
                    )

                self.optimizer_pretrain.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    seg_pred_mr = self.encoder_mr(img_mr)
                    seg_pred_ct = self.encoder_ct(img_ct)   

                    loss = self.dice_loss_fn(
                        torch.cat([seg_pred_mr, seg_pred_ct], dim=0),
                        torch.cat([seg_true_mr, seg_true_ct], dim=0)
                        )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_pretrain)
                self.scaler.update()
                
                train_loss_epoch["total"] += loss.item()
                train_loss_epoch["seg"] += loss.item()

            for k, v in train_loss_epoch.items():
                train_loss_epoch[k] = v / (step + 1)
                self.pretrain_loss[k] = np.append(self.pretrain_loss[k], train_loss_epoch[k])

            wandb.log(
                {f"{phase}_loss": train_loss_epoch["total"]},
                step=epoch + 1
            )
            self.lr_scheduler_pretrain.step(train_loss_epoch["total"])

        elif phase == "train":
            self.AE.encoder.load_state_dict(self.encoder_ct.state_dict())
            self.AE.train()
            self.GSN.train()

            train_loss_epoch = dict(total=0.0, df=0.0, chmf=0.0, norm=0.0, smooth=0.0)
            for step, data_ct in enumerate(self.ct_train_loader):
                img_ct, seg_true_ct, df_true_ct = (
                    data_ct["ct_image"].as_tensor().to(DEVICE),
                    data_ct["ct_label"].as_tensor().to(DEVICE),
                    data_ct["ct_df"].as_tensor().to(DEVICE),
                )
                mesh_true_ct = self.surface_extractor(seg_true_ct)

                self.optimizer_train.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    df_pred_ct, _ = self.AE(img_ct)

                    loss_df = self.mse_loss_fn(df_pred_ct, df_true_ct)                    
                    
                    control_mesh = self.warp_control_mesh(df_pred_ct)
                    subdiv_mesh = self.GSN(control_mesh, self.subdivided_faces.faces_levels)

                    loss_chmf, _ = chamfer_distance(
                        mesh_true_ct.verts_padded(), subdiv_mesh.verts_padded(),
                        point_reduction="mean", batch_reduction="mean")
                    true_points = sample_points_from_meshes(mesh_true_ct, 2 * subdiv_mesh._F)
                    pointcloud_true_ct = Pointclouds(points=true_points)
                    loss_norm = point_mesh_face_distance(subdiv_mesh, pointcloud_true_ct, min_triangle_area=1e-3)
                    loss_smooth = mesh_laplacian_smoothing(subdiv_mesh, method="cotcurv")
                    
                    loss = self.super_params.lambda_[0] * loss_df +\
                        self.super_params.lambda_[1] * loss_chmf +\
                            self.super_params.lambda_[2] * loss_norm +\
                                self.super_params.lambda_[3] * loss_smooth

                self.scaler.scale(loss).backward(retain_graph=True)
                self.scaler.step(self.optimizer_train)
                self.scaler.update()
                
                train_loss_epoch["total"] += loss.item()
                train_loss_epoch["df"] += loss_df.item()
                train_loss_epoch["chmf"] += loss_chmf.item()
                train_loss_epoch["norm"] += loss_norm.item()
                train_loss_epoch["smooth"] += loss_smooth.item()

            for k, v in train_loss_epoch.items():
                train_loss_epoch[k] = v / (step + 1)
                self.train_loss[k] = np.append(self.train_loss[k], train_loss_epoch[k])

            wandb.log(
                {f"{phase}_loss": train_loss_epoch["total"]},
                step=epoch + 1
            )

            self.lr_scheduler_train.step(train_loss_epoch["total"])


    def fine_tune(self, epoch):
        self.AE.encoder.load_state_dict(self.encoder_mr.state_dict())
        self.AE.train()
        self.GSN.train()

        fine_tune_loss_epoch = 0.0
        for step, data_mr in enumerate(self.mr_train_loader):
            img_mr, seg_mr, df_mr = (
                data_mr["mr_image"].as_tensor().to(DEVICE),
                data_mr["mr_label"].as_tensor().to(DEVICE),
                data_mr["mr_df"].as_tensor().to(DEVICE)
            )
            skeleton = self.surface_extractor(seg_mr, use_skeleton=True)

            self.optimizer_fine_tune.zero_grad()
            with torch.autocast(device_type=DEVICE):
                df_pred_mr, _ = self.AE(img_mr)

                loss_df = self.mse_loss_fn(df_pred_mr, df_mr)

                control_mesh = self.warp_control_mesh(df_pred_mr)
                subdiv_mesh = self.GSN(control_mesh, self.subdivided_faces.faces_levels)

                loss_chmf, _ = chamfer_distance(
                    skeleton.verts_padded(), subdiv_mesh.verts_padded(),
                    point_reduction="mean", batch_reduction="mean")
                true_points = sample_points_from_meshes(skeleton, 2 * subdiv_mesh._F)
                pointcloud_true_ct = Pointclouds(points=true_points)
                loss_norm = point_mesh_face_distance(subdiv_mesh, pointcloud_true_ct, min_triangle_area=1e-3)
                loss_smooth = mesh_laplacian_smoothing(subdiv_mesh, method="cotcurv")

                loss = self.super_params.lambda_[0] * loss_df + \
                    self.super_params.lambda_[1] * loss_chmf + \
                        self.super_params.lambda_[2] * loss_norm + \
                            self.super_params.lambda_[3] * loss_smooth

            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.step(self.optimizer_fine_tune)
            self.scaler.update()

            fine_tune_loss_epoch += loss.item()

        fine_tune_loss_epoch = fine_tune_loss_epoch / (step + 1)

        wandb.log(
            {"fine-tune_loss": fine_tune_loss_epoch},
            step=epoch + 1
        )

        self.lr_scheduler_fine_tune.step(fine_tune_loss_epoch)


    def valid(self, epoch, save_on):
        self.encoder_ct.eval(); self.encoder_mr.eval()
        self.AE.eval()
        self.GSN.eval()
        
        # choose the validation loader
        if save_on == "sct":
            modal = "ct"
            self.AE.encoder.load_state_dict(self.encoder_ct.state_dict())
            # valid_loader = self.ct_valid_loader
            valid_loader = self.ct_test_loader
        elif save_on == "cap":
            modal = "mr"
            self.AE.encoder.load_state_dict(self.encoder_mr.state_dict())
            # valid_loader = self.mr_valid_loader
            valid_loader = self.mr_test_loader
        else:
            raise ValueError("Invalid dataset name")

        df_metric_batch_decoder = MSEMetric(reduction="mean_batch")
        msh_metric_batch_decoder = DiceMetric(reduction="mean_batch")

        cached_data = dict()
        choice_case = np.random.choice(len(valid_loader), 1)[0]
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                img, seg_true, df_true = (
                    data[f"{modal}_image"].as_tensor().to(DEVICE),
                    data[f"{modal}_label"].as_tensor().to(DEVICE),
                    data[f"{modal}_df"].as_tensor().to(DEVICE),
                )
                seg_true = torch.logical_or(seg_true == 2, seg_true == 4) if isinstance(self.seg_indices, str) else (seg_true == self.seg_indices)
                # seg_true = (seg_true > 0) if isinstance(self.seg_indices, str) else (seg_true == self.seg_indices)
                seg_true = seg_true.float()

                # evaluate the error between predicted df and the true df
                df_pred, seg_pred = self.AE(img)
                seg_pred = torch.stack([self.post_transform(i).as_tensor() for i in decollate_batch(seg_pred)], dim=0)
                seg_pred = torch.logical_or(seg_pred == 2, seg_pred == 4) if isinstance(self.seg_indices, str) else (seg_pred == self.seg_indices)
                # seg_pred = (seg_pred > 0) if isinstance(self.seg_indices, str) else (seg_pred == self.seg_indices)
                seg_pred = seg_pred.float()
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
                        "seg_true": seg_true[0].cpu(),
                        "subdiv_mesh": subdiv_mesh[0].cpu(),
                    }

        # log dice score
        self.eval_df_score["myo"] = np.append(self.eval_df_score["myo"], df_metric_batch_decoder.aggregate().cpu())
        self.eval_msh_score["myo"] = np.append(self.eval_msh_score["myo"], msh_metric_batch_decoder.aggregate().cpu())
        draw_train_loss(self.pretrain_loss, self.super_params, task_code="stationary", phase="pretrain")
        draw_train_loss(self.train_loss, self.super_params, task_code="stationary", phase="train")
        draw_eval_score(self.eval_df_score, self.super_params, task_code="stationary", module="df")
        draw_eval_score(self.eval_msh_score, self.super_params, task_code="stationary", module="msh")
        wandb.log({
            "train_categorised_loss": wandb.Table(
            columns=[f"pretrain_loss \u2193", f"train_loss \u2193",
                     f"eval_df_error \u2193", f"eval_msh_score \u2191"],
            data=[[
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/pretrain_loss.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/train_loss.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/eval_df_score.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/eval_msh_score.png"),
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
        torch.save(self.AE.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_AutoEncoder.pth")
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
                    "df_X -- true vs pred": wandb.Plotly(ff.create_distplot(
                        [cached_data["df_true"][0].flatten().cpu().numpy(), 
                        cached_data["df_pred"][0].flatten().cpu().numpy()],
                        group_labels=["df_true", "df_pred"],
                        colors=["#EF553B", "#3366CC"],
                        bin_size=0.1
                    )),
                    "df_Y -- true vs pred": wandb.Plotly(ff.create_distplot(
                        [cached_data["df_true"][1].flatten().cpu().numpy(), 
                        cached_data["df_pred"][1].flatten().cpu().numpy()],
                        group_labels=["df_true", "df_pred"],
                        colors=["#EF553B", "#3366CC"],
                        bin_size=0.1
                    )),
                    "df_Z -- true vs pred": wandb.Plotly(ff.create_distplot(
                        [cached_data["df_true"][2].flatten().cpu().numpy(), 
                        cached_data["df_pred"][2].flatten().cpu().numpy()],
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
        # if self.super_params.subdiv_levels > 0:
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

        for data in valid_loader:
            try:
                id = os.path.basename(data[f"{modal}_label"].meta["filename_or_obj"]).replace(".nii.gz", "")
            except:
                id = os.path.basename(data[f"{modal}_label"].meta["filename_or_obj"][0]).replace(".nii.gz", "")
            img = data[f"{modal}_image"].as_tensor().to(DEVICE)

            df_pred, _ = self.AE(img)
            control_mesh = self.warp_control_mesh(df_pred)  
            subdiv_mesh = self.GSN(control_mesh, self.subdivided_faces.faces_levels)
            
            save_obj(
               f"{self.out_dir}/{id}.obj", 
                subdiv_mesh.verts_packed(), subdiv_mesh.faces_packed()
            )


    @torch.no_grad()
    def ablation_study(self, save_on):
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

        assert save_on == "cap", "Ablation study is only available for cap dataset"
        self.AE.encoder.load_state_dict(self.encoder_ct.state_dict())

        for data in self.mr_test_loader:
            try:
                id = os.path.basename(data[f"mr_label"].meta["filename_or_obj"]).replace(".nii.gz", "")
            except:
                id = os.path.basename(data[f"mr_label"].meta["filename_or_obj"][0]).replace(".nii.gz", "")
            img = data[f"mr_image"].as_tensor().to(DEVICE)
            # seg = data[f"mr_label"].as_tensor().to(DEVICE)
            # seg = (seg > 0) if isinstance(self.seg_indices, str) else (seg == self.seg_indices)
            df_true = data[f"mr_df"].as_tensor().to(DEVICE)

            df_pred, _ = self.AE(img)   # (lv, myo, rv)

            # df_surface = torch.where(df_pred[0][2] - 1 < 1e-3, 1, 0)
            # verts, faces = marching_cubes(df_surface.unsqueeze(0).permute(0, 3, 1, 2).float(), 
            #                               isolevel=0.1, return_local_coords=False)
            # # convert coordinates from (z, x, y) to (x, y, z)
            # verts = [vert[:, [1, 0, 2]] for vert in verts]
            # verts = verts[0].cpu().numpy()
            # faces = faces[0].cpu().numpy()
            # seg_coord = np.indices(seg.shape[2:])[:, seg.squeeze().cpu().numpy() > 0].reshape(3, -1).T
            # fig = go.Figure(data=[
            #     go.Scatter3d(
            #         x=seg_coord[:, 0], y=seg_coord[:, 1], z=seg_coord[:, 2],
            #         mode="markers", marker=dict(size=1, color="red")
            #     ),
            #     go.Mesh3d(
            #         x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            #         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            #         color="lightblue", opacity=0.5
            #     )
            # ])
            # fig.write_html("surface_from_dist_field.html")

            os.makedirs(os.path.join(self.out_dir, id), exist_ok=True)
            # save the prediction and true distance field as npy files
            np.save(f"{self.out_dir}/{id}/df_true.npy", df_true[0].cpu().numpy())
            np.save(f"{self.out_dir}/{id}/df_pred.npy", df_pred[0].cpu().numpy())
            # # predicted distance field
            # for axis, i in zip(['X', 'Y', 'Z'], range(3)):
            #     fig = ff.create_distplot(
            #         [df_true[0][i].flatten().cpu().numpy(), df_pred[0][i].flatten().cpu().numpy()],
            #         group_labels=["df_true", "df_pred"],
            #         colors=["#EF553B", "#3366CC"],
            #         show_rug=False, show_hist=False,
            #         bin_size=0.1)
            #     fig.update_layout(
            #         title=f"{axis}-axis",
            #         xaxis_title="Distance Field Value",
            #         yaxis_title="Frequency",
            #         showlegend=False,
            #         template="plotly_white",
            #         margin=dict(l=0, r=0, t=0, b=0),
            #         font=dict(size=12),
            #         width=640, height=360
            #     )
            #     fig.write_image(f"{self.out_dir}/{id}/df_{axis}.png", scale=3)

            # warped + adaptive
            control_mesh = self.warp_control_mesh(df_pred)  
            subdiv_mesh = self.GSN(control_mesh, self.subdivided_faces.faces_levels)
            save_obj(
               f"{self.out_dir}/{id}/adaptive.obj", 
                subdiv_mesh.verts_packed(), subdiv_mesh.faces_packed()
            )

            # unwarped + adaptive
            control_mesh = self.warp_control_mesh(df_true, one_step=True)
            subdiv_mesh = self.GSN(control_mesh, self.subdivided_faces.faces_levels)
            save_obj(
               f"{self.out_dir}/{id}/unwarped.obj", 
                subdiv_mesh.verts_packed(), subdiv_mesh.faces_packed()
            )

            # warped + Loop subdivided
            control_mesh = self.warp_control_mesh(df_pred)
            control_mesh = Trimesh(control_mesh.verts_packed().cpu().numpy(), control_mesh.faces_packed().cpu().numpy())
            for _ in range(2): control_mesh = control_mesh.subdivide_loop()
            save_obj(
               f"{self.out_dir}/{id}/loop_subdivided.obj", 
                torch.tensor(control_mesh.vertices), torch.tensor(control_mesh.faces)
            )

            # unwarped + Loop subdivided
            control_mesh = self.control_mesh.extend(df_true.shape[0]).to(DEVICE)
            control_mesh = Trimesh(control_mesh.verts_packed().cpu().numpy(), control_mesh.faces_packed().cpu().numpy())
            for _ in range(2): control_mesh = control_mesh.subdivide_loop()
            save_obj(
               f"{self.out_dir}/{id}/unwarped_loop_subdivided.obj", 
                torch.tensor(control_mesh.vertices), torch.tensor(control_mesh.faces)
            )

