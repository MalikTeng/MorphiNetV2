"""MR-specific TestBench-compatible masks, metrics, and exports."""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


def find_changing_mask_indices(tensor: torch.Tensor, dim=(-1, -2)) -> torch.Tensor:
    """Return slice indices where non-zero label area changes, matching TestBench."""
    tensor = torch.as_tensor(tensor)
    while tensor.dim() > 3:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be 3D after squeezing singleton dimensions")

    areas = torch.sum(tensor != 0, dim=dim)
    area_diff = torch.diff(areas.to(torch.int64))
    change_indices = torch.where(area_diff != 0)[0] + 1
    results = change_indices.tolist()
    if areas[0] > 0:
        results.insert(0, 0)
    if areas[-1] > 0:
        results.append(tensor.size(0) - 1)
    return torch.tensor(sorted(set(results)), dtype=torch.long, device=tensor.device)


def _as_label_volume(label: torch.Tensor) -> torch.Tensor:
    label = torch.as_tensor(label)
    if label.dim() == 5 and label.shape[0] == 1:
        label = label[0]
    if label.dim() == 4 and label.shape[0] == 1:
        label = label[0]
    return label


def _slice_mask(label_volume: torch.Tensor, fractions=(0.75, 0.5, 0.25)) -> Dict[str, torch.Tensor]:
    indices = find_changing_mask_indices(label_volume, dim=(-1, -2))
    if indices.numel() == 0:
        return {"sax": label_volume.unsqueeze(0)}

    selected = []
    for fraction in fractions:
        pos = min(max(int(indices.numel() * fraction), 0), indices.numel() - 1)
        selected.append(indices[pos])

    masks = []
    for index in selected:
        mask = torch.zeros_like(label_volume, dtype=torch.bool)
        mask[index] = True
        masked = label_volume.clone()
        masked *= mask
        masked[(masked == 0) & mask] = -1
        masks.append(masked)
    return {"sax": torch.stack(masks), "individual_slices": {f"sa{i}": mask for i, mask in enumerate(masks)}}


def create_mr_ground_truth_mask(data, case_id: str, dataset_name: str, super_params) -> Dict[str, torch.Tensor]:
    """Create SAX/LAX masks following the TestBench MR MorphiNet branch."""
    label = torch.as_tensor(data["mr_label"])
    label = torch.where(label == 4, torch.full_like(label, 2), label)
    dataset = (dataset_name or "").lower()

    if dataset == "acdc":
        return _slice_mask(_as_label_volume(label))

    if label.dim() >= 4 and label.shape[0] >= 2:
        sax = _as_label_volume(label[0:1])
        lax = _as_label_volume(label[1:2])
        return {"sax": sax.unsqueeze(0), "lax": lax.unsqueeze(0), "individual_slices": {}}

    return {"sax": _as_label_volume(label).unsqueeze(0), "individual_slices": {}}


def _dice_from_tensors(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred_mask = pred.detach().float() > 0
    true_mask = true.detach().float() == 2
    if pred_mask.shape != true_mask.shape:
        pred_mask = pred_mask.expand_as(true_mask)
    intersection = torch.logical_and(pred_mask, true_mask).sum().item()
    denom = pred_mask.sum().item() + true_mask.sum().item()
    return float((2.0 * intersection) / denom) if denom else 1.0


def compute_mr_mesh_metrics(voxeld_mesh: torch.Tensor, processed_masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute TestBench SAX/LAX Dice overlays for a voxelized MYO mesh."""
    metrics = {}
    for name, mask in processed_masks.items():
        metrics[f"dice_{name}"] = _dice_from_tensors((mask > 0) * voxeld_mesh, mask)
    return metrics


def export_mr_metrics_xlsx(all_mr_metrics: List[Dict], all_case_ids: List[str], export_path: str, dataset_export_name: str) -> str:
    """Export MR SAX/LAX metrics using TestBench-style case rows."""
    os.makedirs(export_path, exist_ok=True)
    rows = []
    for item in all_mr_metrics:
        row = {"case_id": item.get("case_id")}
        row.update(item.get("metrics", {}))
        rows.append(row)

    output_path = os.path.join(export_path, f"{dataset_export_name}_mr_slice_metrics.xlsx")
    pd.DataFrame(rows or [{"case_id": case_id} for case_id in all_case_ids]).to_excel(output_path, index=False)
    return output_path
