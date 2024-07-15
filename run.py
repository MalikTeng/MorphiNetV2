import os, sys, json, glob
from collections import OrderedDict
import trimesh
from trimesh import Trimesh, load
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
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, 
    AsDiscreted,
    KeepLargestConnectedComponentd,
    RemoveSmallObjectsd,
    CropForegroundd,
    Resized,
    Spacingd,
    SpatialPadd,
    EnsureTyped, 
)
from monai.transforms.utils import distance_transform_edt
from monai.utils import set_determinism
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
        self.post_transform = Compose([
            AsDiscreted("pred", argmax=True, allow_missing_keys=True),
            KeepLargestConnectedComponentd("pred", independent=True, allow_missing_keys=True),
            RemoveSmallObjectsd("pred", min_size=8, allow_missing_keys=True),
            Spacingd(["pred", "label"], [2.0, 2.0, 2.0], mode="nearest", allow_missing_keys=True),
            CropForegroundd(["pred", "label"], source_key="label", allow_missing_keys=True),
            Resized(
                ["pred", "label"],
                self.super_params.crop_window_size[0], 
                size_mode="longest", mode="nearest-exact",
                allow_missing_keys=True
                ),

            MaskCTd(["pred"], allow_missing_keys=True),

            SpatialPadd(
                ["pred", "label"],
                self.super_params.crop_window_size[0], 
                method="symmetric", mode="constant",
                allow_missing_keys=True
                ),
            EnsureTyped(["pred", "label"], device=DEVICE, allow_missing_keys=True),
        ])

        self._data_warper(rotation=True)

        # import control mesh (NDC space, [-1, 1]) to compute the subdivision matrix
        template_mesh = load(super_params.control_mesh_dir)
        self.mesh_label = torch.LongTensor(json.load(open(super_params.control_mesh_dir.replace(".obj", ".json"), "r"))).to(DEVICE)
        self.template_mesh = Meshes(
            verts=[torch.tensor(template_mesh.vertices, dtype=torch.float32)], 
            faces=[torch.tensor(template_mesh.faces, dtype=torch.int64)]
            ).to(DEVICE)

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
        train_data = self._remap_abs_path(data_json["train_fold0"], modal, "Tr")
        valid_data = self._remap_abs_path(data_json["validation_fold0"], modal, "Tr")
        test_data = self._remap_abs_path(data_json["test"], modal, "Ts")

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
            out_channels=self.super_params.num_classes + 1,  # include background
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
            out_channels=self.super_params.num_classes + 1,  # include background
            kernel_size=self.super_params.kernel_size, 
            strides=self.super_params.strides,
            upsample_kernel_size=self.super_params.strides[1:], 
            filters=self.super_params.filters, 
            dropout=False,
            deep_supervision=False,
            res_block=True
        ).to(DEVICE)
        self.decoder = ResNet(
            "basic", 
            self.super_params.layers, self.super_params.block_inplanes, 
            n_input_channels=2, # foreground and myocardium
            conv1_t_size=3, conv1_t_stride=1, no_max_pool=True, shortcut_type="B",
            num_classes=2,  # foreground and myocardium
            feed_forward=False, 
        ).to(DEVICE)

        # initialise the subdiv module
        self.subdivided_faces = Subdivision(self.template_mesh, self.super_params.subdiv_levels, mesh_label=self.mesh_label) # create pre-computed subdivision matrix
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
        seg_true = (seg_true == 2).to(torch.float32)    # extract the myocardium
        verts, faces = marching_cubes(
            seg_true.squeeze(1).permute(0, 3, 1, 2), 
            isolevel=0.1,
            return_local_coords=True,
        )
        mesh_true = Meshes(verts, faces)
        mesh_true = taubin_smoothing(mesh_true, 0.77, -0.34, 30)

        return mesh_true


    @torch.no_grad()
    def warp_control_mesh(self, df_preds, t=2.73):
        """
            input:
                df preds: the predicted df.
            return:
                warped control mesh with vertices and faces in NDC space.
        """
        b, *_, d = df_preds.shape
        template_mesh = load(self.super_params.control_mesh_dir)
        template_mesh = Meshes(
            verts=[torch.tensor(template_mesh.vertices, dtype=torch.float32)], 
            faces=[torch.tensor(template_mesh.faces, dtype=torch.int64)]
            ).to(DEVICE).extend(b)
        verts = template_mesh.verts_padded()

        # sample and apply offset in three-stage manner: smooth global -> sharp local
        for stage in range(3):

            df_pred = df_preds[:, 0] if stage < 2 else df_preds[:, 1]      # foreground and myocardium
            # calculate the gradient of the df
            direction = torch.gradient(-df_pred, dim=(1, 2, 3), edge_order=1)
            direction = torch.stack(direction, dim=1)
            direction /= (torch.norm(direction, dim=1, keepdim=True) + 1e-16)
            direction[torch.isnan(direction)] = 0
            direction[torch.isinf(direction)] = 0

            if stage == 0:
                # sampled offset is very fuzzy and apply it directly to the control mesh will break its manifold, so try average the offset and apply it
                offset = torch.stack([2 * (torch.nonzero(df <= 1).to(torch.float32).mean(0) / df.shape[-1] - 0.5) for df in df_pred])[:, [1, 0, 2]] -\
                      verts.mean(1)   # i, j, k -> x, y, z
                offset = offset.unsqueeze(1).expand(-1, template_mesh._V, -1)
                verts += offset

            else:
                # too large the offset that applies close to valve will break the manifold, so try to reduce the offset with equation y = x * e^(-|x|/t) where t is temperature term
                offset = direction * df_pred.unsqueeze(1)
                offset = F.grid_sample(
                    offset.permute(0, 1, 4, 2, 3), 
                    verts.unsqueeze(1).unsqueeze(1),
                    align_corners=True
                ).view(b, 3, -1).transpose(-1, -2)[..., [1, 0, 2]]
                offset = offset * torch.exp(-abs(offset) / t)

                if stage == 1:
                    norm = torch.norm(offset, dim=-1, keepdim=True)
                    offset = offset / (norm + 1e-16) * torch.median(norm, dim=1, keepdim=True).values
                    # transform from NDC space to pixel space
                    verts = (verts / 2 + 0.5) * d
                    verts += offset
                    # transform verts back to NDC space
                    verts = 2 * (verts / d - 0.5)

                else:
                    # transform from NDC space to pixel space
                    verts = (verts / 2 + 0.5) * d
                    verts += offset

        # transform verts back to NDC space and update the control mesh
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
                img_ct, seg_true_ct, df_true_ct = (
                    data_ct["ct_image"].to(DEVICE),
                    data_ct["ct_label"].to(DEVICE),
                    data_ct["ct_df"].as_tensor().to(DEVICE),
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
                    # downsample so that the resolution is the same as CMR data
                    seg_pred_ct = torch.stack([self.post_transform({"pred": i, "label": j})["pred"] 
                                               for i, j in zip(seg_pred_ct, seg_true_ct)], dim=0)
                    seg_pred_ct_ds = F.interpolate(seg_pred_ct.as_tensor(), 
                                                    scale_factor=1 / self.super_params.pixdim[-1], 
                                                    mode="nearest-exact")
                    # compute the distance field
                    foreground = (seg_pred_ct_ds > 0).to(torch.float32)
                    myo = (seg_pred_ct_ds == 2).to(torch.float32)
                    df_pred_ct_ = torch.stack(
                        [distance_transform_edt(i[:, 0]) + distance_transform_edt(1 - i[:, 0]) 
                         for i in [foreground, myo]], 
                         dim=1)
                    # predict the df + residual
                    df_pred_ct = self.decoder(df_pred_ct_)

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
                img_ct, seg_true_ct = (
                    data_ct["ct_image"].to(DEVICE),
                    data_ct["ct_label"].to(DEVICE),
                )
                seg_true_ct_ = torch.stack([self.post_transform({"label": i})["label"] for i in seg_true_ct], dim=0)
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
                    seg_pred_ct = torch.stack([self.post_transform({"pred": i, "label": j})["pred"] 
                                               for i, j in zip(seg_pred_ct, seg_true_ct)], dim=0)
                    seg_pred_ct_ds = F.interpolate(seg_pred_ct.as_tensor(), 
                                                    scale_factor=1 / self.super_params.pixdim[-1], 
                                                    mode="nearest-exact")
                    foreground = (seg_pred_ct_ds > 0).to(torch.float32)
                    myo = (seg_pred_ct_ds == 2).to(torch.float32)
                    df_pred_ct_ = torch.stack(
                        [distance_transform_edt(i[:, 0]) + distance_transform_edt(1 - i[:, 0]) 
                         for i in [foreground, myo]], 
                         dim=1)
                    df_pred_ct = self.decoder(df_pred_ct_)
                    
                    template_mesh = self.warp_control_mesh(df_pred_ct.detach())
                    level_outs = self.GSN(template_mesh, self.subdivided_faces.faces_levels)

                    loss_chmf, loss_norm, loss_smooth = 0.0, 0.0, 0.0
                    for l, subdiv_mesh in enumerate(level_outs):
                        loss_chmf += chamfer_distance(
                            mesh_true_ct.verts_padded(), subdiv_mesh.verts_padded(),
                            point_reduction="mean", batch_reduction="mean")[0]
                        # true_points = sample_points_from_meshes(mesh_true_ct, 2 * subdiv_mesh._F)
                        # pointcloud_true_ct = Pointclouds(points=true_points)
                        # loss_norm += point_mesh_face_distance(subdiv_mesh, pointcloud_true_ct, min_triangle_area=1e-3)
                        pointcloud_pred_ct = subdiv_mesh.verts_packed()[subdiv_mesh.faces_packed()].mean(1).unsqueeze(0)
                        loss_norm += surface_crossing_loss(subdiv_mesh, pointcloud_pred_ct.detach(), self.subdivided_faces.labels_levels[l],
                                                           min_triangle_area=1e-3)
                        loss_smooth += mesh_laplacian_smoothing(subdiv_mesh, method="uniform")
                    
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

        elif phase == "ndf":
            self.NDF.train()

            train_loss_epoch = dict(total=0.0, ndf=0.0)
            for step, data_mr in enumerate(self.mr_train_loader):
                img_mr, seg_true_mr = (
                    data_mr["mr_image"].to(DEVICE),
                    data_mr["mr_label"].to(DEVICE),
                )
                batch = data_mr["mr_batch"].item()
                seg_true_mr = seg_true_mr.unflatten(0, (batch, -1)).swapaxes(1, 2)

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
                    seg_pred_mr = seg_pred_mr.unflatten(0, (batch, -1)).swapaxes(1, 2)
                    seg_pred_mr = torch.stack([self.post_transform({"pred": i, "label": j})["pred"] 
                                               for i, j in zip(seg_pred_mr, seg_true_mr)], dim=0)
                    seg_pred_mr_ds = F.interpolate(seg_pred_mr.as_tensor(), 
                                                    scale_factor=1 / self.super_params.pixdim[-1], 
                                                    mode="nearest-exact")
                    foreground = (seg_pred_mr_ds > 0).to(torch.float32)
                    myo = (seg_pred_mr_ds == 2).to(torch.float32)
                    df_pred_mr_ = torch.stack(
                        [distance_transform_edt(i[:, 0]) + distance_transform_edt(1 - i[:, 0]) 
                         for i in [foreground, myo]], 
                         dim=1)
                    df_pred_mr = self.decoder(df_pred_mr_)
                    
                    template_mesh = self.warp_control_mesh(df_pred_mr).detach()
                    try:
                        # method 1: NDF applied right after warping the control mesh
                        ndf_verts = self.NDF(template_mesh.verts_padded()[0], end_time=1, step=batch-1, invert=False)
                        loss_ndf = self.l1_loss_fn(ndf_verts, template_mesh.verts_padded())
                    except AssertionError:
                        # print("NDF failed to converge.")
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
        if self.super_params._4d.lower() == 'y':
            self.NDF.eval()
        
        # save model
        ckpt_weight_path = os.path.join(self.ckpt_dir, "trained_weights")
        os.makedirs(ckpt_weight_path, exist_ok=True)
        torch.save(self.encoder_ct.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_UNet_CT.pth")
        torch.save(self.encoder_mr.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_UNet_MR.pth")
        torch.save(self.decoder.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_ResNet.pth")
        torch.save(self.GSN.state_dict(), f"{ckpt_weight_path}/{epoch + 1}_GSN.pth")
        if self.super_params._4d.lower() == 'y':
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
                seg_data = [self.post_transform({"pred": i, "label": j}) for i, j in zip(seg_pred, seg_true)]
                seg_pred = torch.stack([i["pred"] for i in seg_data], dim=0)
                seg_pred_ds = F.interpolate(seg_pred.as_tensor(), 
                                                scale_factor=1 / self.super_params.pixdim[-1], 
                                                mode="nearest-exact")
                foreground = (seg_pred_ds > 0).to(torch.float32)
                myo = (seg_pred_ds == 2).to(torch.float32)
                df_pred_ = torch.stack(
                    [distance_transform_edt(i[:, 0]) + distance_transform_edt(1 - i[:, 0]) 
                        for i in [foreground, myo]], 
                        dim=1)
                df_pred = self.decoder(df_pred_)
                df_metric_batch_decoder(df_pred, df_true)

                # evaluate the error between subdivided mesh and the true segmentation
                template_mesh = self.warp_control_mesh(df_pred) 

                if save_on == "cap" and self.super_params._4d == 'y':
                    # method 1: NDF applied right after warping the control mesh
                    ndf_verts = self.NDF(template_mesh.verts_padded()[0], end_time=1, step=batch-1, invert=False) 
                    template_mesh = template_mesh.update_padded(ndf_verts)

                subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels)[-1]
                
                # if save_on == "cap" and self.super_params._4d == 'y':
                #     # method 2: NDF applied after the GSN
                #     ndf_verts = self.NDF(subdiv_mesh.verts_padded()[0], end_time=1, step=batch-1, invert=False)
                #     subdiv_mesh = subdiv_mesh.update_padded(ndf_verts)
                
                voxeld_mesh = torch.cat([
                    self.rasterizer(
                        pred_mesh.verts_padded(), pred_mesh.faces_padded())
                    for pred_mesh in subdiv_mesh
                    ], dim=0)     
                seg_true = torch.stack([i["label"] for i in seg_data], dim=0)
                seg_true = (seg_true.as_tensor() == 2).to(torch.float32)
                msh_metric_batch_decoder(voxeld_mesh, seg_true)

                if step == choice_case:
                    df_true = df_true[:, 1].unsqueeze(1)
                    df_pred = df_pred[:, 1].unsqueeze(1)
                    seg_pred = (seg_pred == 2).to(torch.float32)
                    seg_pred_ds = (seg_pred_ds == 2).to(torch.float32)

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
        draw_train_loss(
            self.ndf_loss if self.super_params._4d.lower() == 'y' else self.gsn_loss, 
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
            if self.super_params._4d.lower() == 'y':
                torch.save(self.NDF.state_dict(), f"{ckpt_weight_path}/best_NDF.pth")
            # save the subdivided_faces.faces_levels as pth file
            for level, faces in enumerate(self.subdivided_faces.faces_levels):
                torch.save(faces, f"{ckpt_weight_path}/{epoch+1}_subdivided_faces_l{level}.pth")
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
                        mesh_pred=self.template_mesh.clone().cpu(),
                        )),
                    "seg_true_ds vs df_pred": wandb.Plotly(draw_plotly(
                        seg_true=cached_data["seg_true_ds"], 
                        df_pred=cached_data["df_pred"],
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
            template_mesh = self.warp_control_mesh(df_pred)  
            subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels)

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

        for i, data in enumerate(self.mr_test_loader):
            id = os.path.basename(self.mr_test_loader.dataset.data[i]["mr_label"]).replace(".nii.gz", '').replace(".seg.nrrd", '')
            id = id.split('-')[0] + '-ED'
            img = data[f"mr_image"][:1].as_tensor().to(DEVICE)
            df_true = data[f"mr_df"].as_tensor().to(DEVICE)

            df_pred, _ = self.AE(img)   # (lv, myo, rv)

            os.makedirs(os.path.join(self.out_dir, id), exist_ok=True)
            # save the prediction and true distance field as npy files
            np.save(f"{self.out_dir}/{id}/df_true.npy", df_true[0].cpu().numpy())
            np.save(f"{self.out_dir}/{id}/df_pred.npy", df_pred[0].cpu().numpy())

            # warped + adaptive
            template_mesh = self.warp_control_mesh(df_pred)  
            subdiv_mesh = self.GSN(template_mesh, self.subdivided_faces.faces_levels)
            # smoothing the generated mesh
            subdiv_mesh = taubin_smoothing(subdiv_mesh, 0.77, -0.34, 30)
            save_obj(
               f"{self.out_dir}/{id}/adaptive.obj", 
                subdiv_mesh.verts_packed(), subdiv_mesh.faces_packed()
            )

            # warped + Loop subdivided
            template_mesh = self.warp_control_mesh(df_pred)
            template_mesh = Trimesh(template_mesh.verts_packed().cpu().numpy(), template_mesh.faces_packed().cpu().numpy())
            for _ in range(2): template_mesh = template_mesh.subdivide_loop()
            save_obj(
               f"{self.out_dir}/{id}/loop_subdivided.obj", 
                torch.tensor(template_mesh.vertices), torch.tensor(template_mesh.faces)
            )

            # unwarped + Loop subdivided
            template_mesh = self.template_mesh.to(DEVICE)
            template_mesh = Trimesh(template_mesh.verts_packed().cpu().numpy(), template_mesh.faces_packed().cpu().numpy())
            for _ in range(2): template_mesh = template_mesh.subdivide_loop()
            save_obj(
            f"{self.out_dir}/{id}/unwarped_loop_subdivided.obj", 
                torch.tensor(template_mesh.vertices), torch.tensor(template_mesh.faces)
            )

            # control mesh
            save_obj(
                f"{self.out_dir}/{id}/template_mesh.obj", 
                self.template_mesh.verts_packed(), self.template_mesh.faces_packed()
            )

