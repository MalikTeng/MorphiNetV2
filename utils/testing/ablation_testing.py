"""TestBench-style array metrics used by MorphiNet ablation exports."""

from typing import Dict

import numpy as np
from scipy.spatial.distance import directed_hausdorff

LABELS = {"LV": 1, "MYO": 2, "RV": 3}


def _dice(pred: np.ndarray, true: np.ndarray) -> float:
    intersection = np.logical_and(pred, true).sum()
    denom = pred.sum() + true.sum()
    return float((2.0 * intersection) / denom) if denom else 1.0


def _hausdorff(pred: np.ndarray, true: np.ndarray) -> float:
    pred_points = np.argwhere(pred)
    true_points = np.argwhere(true)
    if len(pred_points) == 0 or len(true_points) == 0:
        return 0.0
    return float(max(directed_hausdorff(pred_points, true_points)[0], directed_hausdorff(true_points, pred_points)[0]))


def compute_ablation_dice_scores_all_labels(pred_arr: np.ndarray, true_arr: np.ndarray, num_classes: int, inference=None) -> Dict[str, float]:
    """Compute per-label Dice using the same LV/MYO/RV label IDs as TestBench."""
    return {name: _dice(pred_arr == label, true_arr == label) for name, label in LABELS.items() if label < num_classes}


def compute_ablation_hausdorff_distances(pred_arr: np.ndarray, true_arr: np.ndarray, num_classes: int, inference=None) -> Dict[str, float]:
    """Compute per-label symmetric Hausdorff using TestBench label IDs."""
    return {name: _hausdorff(pred_arr == label, true_arr == label) for name, label in LABELS.items() if label < num_classes}


def compute_volume_differences(before_arr: np.ndarray, after_arr: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Return structured volume deltas expected by the MorphiNet XLSX exporter."""
    diffs: Dict[str, Dict[str, float]] = {}
    for name, label in LABELS.items():
        before_count = int(np.sum(before_arr == label))
        after_count = int(np.sum(after_arr == label))
        absolute_diff = abs(after_count - before_count)
        percentage_diff = (absolute_diff / before_count * 100.0) if before_count else 0.0
        diffs[name] = {
            "absolute_diff": absolute_diff,
            "percentage_diff": float(percentage_diff),
            "before_count": before_count,
            "after_count": after_count,
        }
    return diffs


def extract_phase_from_case_id(case_id: str, dataset_name: str) -> str:
    """Infer ED/ES labels from ACDC/CAP-style case IDs used by TestBench."""
    lowered = case_id.lower()
    if lowered.endswith("_ed") or "-ed" in lowered:
        return "ED"
    if lowered.endswith("_es") or "-es" in lowered:
        return "ES"
    if "frame01" in lowered or "frame001" in lowered:
        return "ED"
    return "ES" if "frame" in lowered else "unknown"
