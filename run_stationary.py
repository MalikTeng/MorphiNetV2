import os, sys, json
from collections import OrderedDict
from itertools import chain
from trimesh import load_mesh
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import save_obj, IO
from pytorch3d.ops import sample_points_from_meshes, taubin_smoothing
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
from pytorch3d.ops.marching_cubes import marching_cubes
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import compute_hausdorff_distance, MeanIoU, DiceMetric, MSEMetric
from monai.transforms import (
    Compose, 
    SpatialPad,
    EnsureType, 
    Rotate90,
    KeepLargestConnectedComponent,
    Invertd,
)
from monai.utils import set_determinism
from midvoxio.voxio import vox_to_arr
import wandb

import plotly.io as pio
import plotly.graph_objects as go

from data import *
from utils import *
from model.networks import *


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

        self.ckpt_dir = os.path.join(super_params.ckpt_dir, "stationary", super_params.run_id)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_dir, "trained_weights"), exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_dir, "valid_subdiv_meshes"), exist_ok=True)
        if is_training:
            self.sdf_predict_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "seg", "sdf"]}
            )
            self.subdiv_loss = OrderedDict(
                {k: np.asarray([]) for k in ["total", "chmf", "norm", "smooth"]}
            )
            self.fine_tune_loss = {"hd": np.asarray([])}
            self.eval_sdf_score = OrderedDict(
                {k: np.asarray([]) for k in ["myo"]}
            )
            self.eval_msh_score = self.eval_sdf_score.copy()
            self.best_eval_score = 0
        else:
            self.out_dir = os.path.join(super_params.out_dir, "stationary", super_params.run_id)
            os.makedirs(self.out_dir, exist_ok=True)

        self.post_transform = Compose([
            SpatialPad(self.super_params.crop_window_size, mode="minimum"),
            KeepLargestConnectedComponent(is_onehot=False),
            Rotate90(spatial_axes=(1, 2), k=-1),
            EnsureType(data_type="tensor", dtype=torch.float, device=DEVICE),
            ])

        with open(super_params.mr_json_dir, "r") as f:
            mr_train_transform, mr_valid_transform = self._prepare_transform(["mr_image", "mr_label"], "mr")
            mr_train_ds, mr_valid_ds, mr_test_ds = self._prepare_dataset(
                json.load(f), "mr", mr_train_transform, mr_valid_transform
            )
            self.mr_train_loader, self.mr_valid_loader, self.mr_test_loader = self._prepare_dataloader(
                mr_train_ds, mr_valid_ds, mr_test_ds
            )

        self.inverse_transform = Invertd(
                keys="mr_label_origin", transform=mr_train_transform, 
                nearest_interp=False, to_tensor=True
            )
        
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
            faces=[torch.tensor(control_mesh.faces, dtype=torch.int64)]).to(DEVICE)
        # define the indices of sdf and segmentation so that loss is calculated based on matching label class
        if "lv" in super_params.control_mesh_dir.lower():
            self.sdf_indices = 0
            self.seg_indices = 1
        elif "myo" in super_params.control_mesh_dir.lower():
            self.sdf_indices = 1        # index of channel
            self.seg_indices = [2, 4]   # index of class
        elif "rv" in super_params.control_mesh_dir.lower():
            self.sdf_indices = 2
            self.seg_indices = 3
        else:
            raise ValueError("Invalid control_mesh_dir")

        self._prepare_modules()
        self._prepare_optimiser()


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
                "mr_slice_info": os.path.join(self.super_params.mr_data_dir, "slice_info", os.path.basename(d["label"])).replace(".nii.gz", "-info.json"),
            } for d in data_list]
        elif modal == "ct":
            return [{
                "ct_image": os.path.join(self.super_params.ct_data_dir, "imagesTr", os.path.split(d["image"])[-1]),
                "ct_label": os.path.join(self.super_params.ct_data_dir, "labelsTr", os.path.split(d["label"])[-1]),
            } for d in data_list]
        

    def _prepare_dataset(self, data_json, modal, train_transform, valid_transform):
        # WARNING: must remove the array slice for training !!!
        train_data = self._remap_abs_path(data_json["train_fold0"], modal)[:5]
        valid_data = self._remap_abs_path(data_json["validation_fold0"], modal)[:5]
        test_data = self._remap_abs_path(data_json["test"], modal)[:5]

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
        train_loader = DataLoader(
            train_ds, batch_size=self.super_params.batch_size,
            shuffle=True, num_workers=self.num_workers,
            )
        val_loader = DataLoader(
            valid_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers,
            )
        test_loader = DataLoader(
            test_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers,
            )
        
        return train_loader, val_loader, test_loader


    def _prepare_modules(self):
        # initialise the sdf module
        self.pretext_mr = ResEncoder(
            in_channels=1, out_channels=self.super_params.num_classes + 1,  # include background
            init_filters=self.super_params.init_filters,
            act='relu', norm='batch',
            num_init_blocks=self.super_params.num_init_blocks
        ).to(DEVICE)
        self.sdf_module = SDFNet(
            encoder=ResEncoder(
                in_channels=1, out_channels=self.super_params.num_classes + 1,
                init_filters=self.super_params.init_filters,
                act='relu', norm='batch',
                num_init_blocks=self.super_params.num_init_blocks
            ),
            decoder=ResDecoder(
                in_channels=self.super_params.init_filters * 2 ** (len(self.super_params.num_init_blocks) - 1), 
                out_channels=self.super_params.num_classes-1,
                act='relu', norm='batch',
            )).to(DEVICE)
        if self.super_params.pre_trained_mr_module_dir is not None and self.super_params.pre_trained_sdf_module_dir is not None:
            self.pretext_mr.load_state_dict(torch.load(self.super_params.pre_trained_mr_module_dir))
            self.sdf_module.load_state_dict(torch.load(self.super_params.pre_trained_sdf_module_dir))
        else:
            print("WARN: No pre-trained SDFNet module is found")

        # initialise the GSN module
        self.gsn_module = GSN(
            self.control_mesh, 
            hidden_features=self.super_params.hidden_features_gsn, 
            num_layers=self.super_params.subdiv_levels,
        ).to(DEVICE)
        

    def _prepare_optimiser(self):
        self.loss_fn_seg = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            # squared_pred=True,
            )
        self.loss_fn_sdf = nn.MSELoss()
        self.loss_fn_silhouette = slice_silhouette_loss

        # initialise the optimiser for sdf_predict
        self.optimizer_sdf = torch.optim.Adam(
            chain(self.pretext_mr.parameters(), self.sdf_module.parameters()), 
            lr=self.super_params.lr)
        self.lr_scheduler_sdf = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_sdf, mode="min", factor=0.1, patience=5, verbose=True,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )

        # initialise the optimiser for subdiv
        self.optimizer_gsn = torch.optim.Adam(
            self.gsn_module.parameters(), 
            lr=self.super_params.lr
        )
        self.lr_scheduler_gsn = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gsn, mode="min", factor=0.5, patience=10, verbose=True,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=0, min_lr=1e-6, eps=1e-8
            )
        
        # initialise the optimiser for fine-tune
        self.optimizer_fine_tune = torch.optim.Adam(
            self.gsn_module.parameters(), 
            lr=self.super_params.lr
        )
        self.lr_scheduler_fine_tune = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fine_tune, mode="min", factor=0.1, patience=5, verbose=True,
            threshold=1e-2, threshold_mode="rel", 
            cooldown=self.super_params.max_epochs//50, min_lr=1e-6, eps=1e-8
            )
        
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
        seg_true = (seg_true == self.seg_indices[0]) | (seg_true == self.seg_indices[1]) if isinstance(self.seg_indices, list) else (seg_true == self.seg_indices)
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

        return mesh_true


    def warp_control_mesh(self, sdf_pred):
        """
            input:
                control_mesh: the control mesh in NDC space [-1, 1].
                sdf_pred: the predicted sdf.
            return:
                warped control mesh with vertices and faces in NDC space.
        """
        sdf_pred = sdf_pred[:, self.sdf_indices].unsqueeze(1)
        control_mesh = self.control_mesh.extend(sdf_pred.shape[0])
        # calculate the gradient of the sdf
        direction = torch.cat(torch.gradient(sdf_pred, dim=(2, 3, 4)), dim=1)

        # get the offset by the magnitude and inverse direction of the sdf (N, D, V) and grid_sample
        direction /= torch.norm(direction, dim=1, keepdim=True) * -1
        offset = direction * sdf_pred
        # get the offset using grid_sample
        offset = F.grid_sample(
            offset, 
            torch.flip(control_mesh.verts_padded(), dims=(-1,)).unsqueeze(1).unsqueeze(1),
            # x, y, z -> i, j, k
            align_corners=True
        ).view(sdf_pred.shape[0], 3, -1).permute(0, 2, 1)

        # apply offset to the vertices
        control_mesh.scale_verts_(sdf_pred.shape[-1] / 2)
        control_mesh.offset_verts_(offset.reshape(-1, 3))

        # rescale back to NDC space
        control_mesh.scale_verts_(2 / sdf_pred.shape[-1])

        return control_mesh


    def train_iter(self, epoch, phase):
        if phase == "sdf_predict":
            self.pretext_mr.train()
            self.sdf_module.train()

            train_loss_epoch = dict(
                total=0.0, seg=0.0, sdf=0.0
            )
            for step, (data_ct, data_mr) in enumerate(zip(self.ct_train_loader, self.mr_train_loader)):
                img_ct, seg_true_ct, sdf_true_ct = (
                    data_ct["ct_image"].as_tensor().to(DEVICE),
                    data_ct["ct_label"].as_tensor().to(DEVICE),
                    data_ct["ct_sdf"].as_tensor().to(DEVICE),
                    )
                img_mr, seg_true_mr = (
                    data_mr["mr_image"].as_tensor().to(DEVICE),
                    data_mr["mr_label"].as_tensor().to(DEVICE),
                    )

                self.optimizer_sdf.zero_grad()
                with torch.autocast(device_type=DEVICE):
                    _, seg_pred_mr = self.pretext_mr(img_mr)
                    sdf_pred_ct, seg_pred_ct = self.sdf_module(img_ct)

                    loss_seg = self.loss_fn_seg(
                        torch.cat([seg_pred_mr, seg_pred_ct], dim=0), 
                        torch.cat([seg_true_mr, seg_true_ct], dim=0))
                    loss_sdf = self.loss_fn_sdf(sdf_pred_ct, sdf_true_ct)

                    loss = loss_seg + loss_sdf

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_sdf)
                self.scaler.update()
                
                train_loss_epoch["total"] += loss.item()
                train_loss_epoch["seg"] += loss_seg.item()
                train_loss_epoch["sdf"] += loss_sdf.item()

            for k, v in train_loss_epoch.items():
                train_loss_epoch[k] = v / (step + 1)
                self.sdf_predict_loss[k] = np.append(self.sdf_predict_loss[k], train_loss_epoch[k])

            wandb.log(
                {f"{phase}_loss": train_loss_epoch["total"]},
                step=epoch + 1
            )
            self.lr_scheduler_sdf.step(train_loss_epoch["total"])

        elif phase == "subdiv":
            self.sdf_module.eval()
            self.gsn_module.train()

            train_loss_epoch = dict(
                total=0.0, chmf=0.0, norm=0.0, smooth=0.0
            )
            for step, data_ct in enumerate(self.ct_train_loader):
                img_ct, seg_true_ct = (
                    data_ct["ct_image"].as_tensor().to(DEVICE),
                    data_ct["ct_label_origin"].as_tensor().to(DEVICE),
                )
                mesh_true_ct = self.surface_extractor(seg_true_ct)

                self.optimizer_gsn.zero_grad()
                sdf_pred_ct, _ = self.sdf_module(img_ct)

                # TODO: test on warping the control mesh
                # control_mesh = self.warp_control_mesh(sdf_pred_ct.detach())
                control_mesh = self.control_mesh.extend(sdf_pred_ct.shape[0])
                subdiv_mesh = self.gsn_module(control_mesh)

                true_points, true_normals = sample_points_from_meshes(mesh_true_ct, self.super_params.point_limit, return_normals=True)
                subdiv_points, subdiv_normals = sample_points_from_meshes(subdiv_mesh, self.super_params.point_limit, return_normals=True)
                # TODO: try BeamGapLoss in gsn.py
                loss_chmf, loss_norm = chamfer_distance(
                    subdiv_points, true_points,
                    x_normals=subdiv_normals, y_normals=true_normals,
                    point_reduction="mean", batch_reduction="sum"
                    )
                loss_smooth = mesh_laplacian_smoothing(subdiv_mesh, method="cotcurv")
                loss = self.super_params.lambda_[0] * loss_chmf + \
                    self.super_params.lambda_[1] * loss_norm + \
                        self.super_params.lambda_[2] * loss_smooth

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_gsn)
                self.scaler.update()
                
                train_loss_epoch["total"] += loss.item()
                train_loss_epoch["chmf"] += loss_chmf.item()
                train_loss_epoch["norm"] += loss_norm.item()
                train_loss_epoch["smooth"] += loss_smooth.item()

            for k, v in train_loss_epoch.items():
                train_loss_epoch[k] = v / (step + 1)
                self.subdiv_loss[k] = np.append(self.subdiv_loss[k], train_loss_epoch[k])

            wandb.log(
                {f"{phase}_loss": train_loss_epoch["total"]},
                step=epoch + 1
            )

            self.lr_scheduler_gsn.step(train_loss_epoch["total"])


    def fine_tune(self, epoch):
        self.sdf_module.eval()
        self.gsn_module.train()

        fine_tune_loss_epoch = {"hd": 0.0}
        for step, data_mr in enumerate(self.mr_train_loader):
            img_mr, slice_info_mr = (
                data_mr["mr_image"].as_tensor().to(DEVICE),
                data_mr["mr_slice_info"])
            seg_true_origin_mr = [self.inverse_transform(i)["mr_label_origin"] for i in decollate_batch(data_mr)]

            self.optimizer_fine_tune.zero_grad()
            sdf_pred_mr, _ = self.sdf_module(img_mr)
            control_mesh = self.warp_control_mesh(sdf_pred_mr.detach())
            subdiv_mesh = self.gsn_module(control_mesh)

            loss = self.loss_fn_silhouette(
                subdiv_mesh, seg_true_origin_mr, self.super_params.crop_window_size[0] // self.super_params.pixdim[0], slice_info_mr, self.seg_indices, DEVICE)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_fine_tune)
            self.scaler.update()

            fine_tune_loss_epoch["hd"] += loss.item()

        fine_tune_loss_epoch["hd"] = fine_tune_loss_epoch["hd"] / (step + 1)

        wandb.log(
            {"fine-tune_loss": fine_tune_loss_epoch["hd"]},
            step=epoch + 1
        )

        self.lr_scheduler_fine_tune.step(fine_tune_loss_epoch["hd"])


    def valid(self, epoch, dataset):
        self.sdf_module.eval()
        self.gsn_module.eval()
        
        # choose the validation loader
        if dataset.lower() == "sct":
            valid_loader = self.ct_valid_loader
            modal = "ct"
        elif dataset.lower() == "cap":
            valid_loader = self.mr_valid_loader
            modal = "mr"
        else:
            raise ValueError("Invalid dataset name")

        sdf_metric_batch_decoder = MSEMetric(reduction="mean_batch")
        msh_metric_batch_decoder = DiceMetric(reduction="mean_batch")

        cached_data = dict()
        choice_case = np.random.choice(len(valid_loader), 1)[0]
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                try:
                    id = os.path.basename(data[f"{modal}_label"].meta["filename_or_obj"]).strip(".nii.gz")
                except TypeError:
                    id = os.path.basename(data[f"{modal}_label"].meta["filename_or_obj"][0]).strip(".nii.gz")
                img, sdf_true, seg_true = (
                    data[f"{modal}_image"].as_tensor().to(DEVICE),
                    data[f"{modal}_sdf"].as_tensor().to(DEVICE),
                    data[f"{modal}_label_origin"].as_tensor().to(DEVICE),
                )

                seg_true = (seg_true == self.seg_indices[0]) | (seg_true == self.seg_indices[1]) if isinstance(self.seg_indices, list) else (seg_true == self.seg_indices)
                seg_true = seg_true.float()

                sdf_pred, _ = self.sdf_module(img)
                sdf_metric_batch_decoder(sdf_pred, sdf_true)

                # TODO: test on warping the control mesh
                # control_mesh = self.warp_control_mesh(sdf_pred)
                control_mesh = self.control_mesh.extend(sdf_pred.shape[0])
                subdiv_mesh = self.gsn_module(control_mesh)

                # rescale and translate the subdivided mesh to the original space
                subdiv_mesh.scale_verts_(self.super_params.crop_window_size[0] // 2)
                subdiv_mesh.offset_verts_(
                    torch.tensor([self.super_params.crop_window_size[0] // 2] * 3, dtype=torch.float32, device=DEVICE)
                )

                # save subdivided mesh as .obj files for voxelisation
                save_obj(f"{self.ckpt_dir}/valid_subdiv_meshes/{id}.obj", 
                         subdiv_mesh.verts_packed(), subdiv_mesh.faces_packed())
                # voxelize and fill the surface mesh
                os.system(f"utils/cuda_voxelizer/build/cuda_voxelizer -f {self.ckpt_dir}/valid_subdiv_meshes/{id}.obj -s {self.super_params.crop_window_size[0]} -o vox -solid")
                # load the voxelised meshes
                voxeld_mesh = vox_to_arr(
                    f"{self.ckpt_dir}/valid_subdiv_meshes/{id}.obj_128.vox").mean(-1)[None]
                voxeld_mesh = self.post_transform(np.round(voxeld_mesh)).unsqueeze(0)
                
                # evaluate the error between seudo mesh and predicted mesh
                msh_metric_batch_decoder(voxeld_mesh, seg_true)

                if step == choice_case:
                    cached_data = {
                        "seg_true": seg_true.cpu(),
                        "subdiv_mesh": subdiv_mesh.cpu(),
                    }

        # log dice score
        self.eval_sdf_score["myo"] = np.array([sdf_metric_batch_decoder.aggregate().cpu()])
        self.eval_msh_score["myo"] = np.array([msh_metric_batch_decoder.aggregate().cpu()])
        draw_train_loss(self.sdf_predict_loss, self.super_params, task_code="stationary", phase="sdf_predict")
        draw_train_loss(self.subdiv_loss, self.super_params, task_code="stationary", phase="subdiv")
        draw_eval_score(self.eval_sdf_score, self.super_params, task_code="stationary", module="sdf")
        draw_eval_score(self.eval_msh_score, self.super_params, task_code="stationary", module="msh")
        wandb.log({
            "train_categorised_loss": wandb.Table(
            columns=[f"sdf_predict_loss \u2193", f"subdiv_loss \u2193",
                     f"eval_sdf_score \u2193", f"eval_msh_score \u2191"],
            data=[[
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/sdf_predict_loss.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/subdiv_loss.png"),
                wandb.Image(f"{self.super_params.ckpt_dir}/stationary/{self.super_params.run_id}/eval_sdf_score.png"),
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
        torch.save(self.pretext_mr.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_ResNet.pth")
        torch.save(self.sdf_module.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_SDFNet.pth")
        torch.save(self.gsn_module.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_GSN.pth")

        # save visualization when the eval score is the best
        if eval_score_epoch > self.best_eval_score:
            self.best_eval_score = eval_score_epoch
            # WARNING: this part is very time-consuming, please comment it if you don"t need it
            wandb.log(
                {
                    "seg_label vs mesh_pred": wandb.Plotly(draw_plotly(
                        labels=cached_data["seg_true"], 
                        pred_meshes=cached_data["subdiv_mesh"]
                        ))
                },
                step=epoch + 1
            )
         

    def test(self, dataset):
        # load modules
        self.pretext_mr.load_state_dict(
            torch.load(os.path.join(f"{self.ckpt_dir}/trained_weights", f"{self.super_params.best_epoch}_ResNet.pth"))
        )
        self.sdf_module.load_state_dict(
            torch.load(os.path.join(f"{self.ckpt_dir}/trained_weights", f"{self.super_params.best_epoch}_SDFNet.pth"))
        )
        self.gsn_module.load_state_dict(
            torch.load(os.path.join(f"{self.ckpt_dir}/trained_weights", f"{self.super_params.best_epoch}_GSN.pth"))
        )
        self.pretext_mr.eval()
        self.sdf_module.eval()
        self.gsn_module.eval()

        if dataset.lower() == "scotheart":
            val_loader = self.val_loader_ct
            modal = "ct"
        elif dataset.lower() == "cap":
            val_loader = self.val_loader_mr
            self.sdf_module.encoder_ = self.pretext_mr
            modal = "mr"
        else:
            raise ValueError("Invalid dataset name")

        # testing
        with torch.no_grad():
            for val_data in val_loader:
                img = val_data[f"{modal}_image"].as_tensor().to(DEVICE)

                sdf_pred, _ = self.sdf_module(img)
                control_mesh = self.warp_control_mesh(sdf_pred)
                save_obj(
                    f"{self.out_dir}/{modal}_pred-sdf_module.obj", 
                    control_mesh.verts_packed(), control_mesh.faces_packed()
                )

                subdiv_mesh = self.gsn_module(control_mesh)
                save_obj(
                    f"{self.out_dir}/{modal}_pred-gsn_module.obj", 
                    subdiv_mesh.verts_packed(), subdiv_mesh.faces_packed()
                )

