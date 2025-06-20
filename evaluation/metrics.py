import torch
import numpy as np
from monai.metrics import DiceMetric, MSEMetric
from pytorch3d.loss import chamfer_distance


class MorphiNetMetrics:
    """Handles metric computation for MorphiNet evaluation."""
    
    def __init__(self, num_classes=5, include_background=False):
        """
        Initialize metrics.
        
        Args:
            num_classes: Number of segmentation classes
            include_background: Whether to include background in dice computation
        """
        self.num_classes = num_classes
        
        # Initialize MONAI metrics
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="mean_batch"
        )
        
        self.mse_metric = MSEMetric(reduction="mean_batch")
        
        # Tracking arrays
        self.reset_tracking()
    
    def reset_tracking(self):
        """Reset metric tracking arrays."""
        self.dice_scores = []
        self.mse_scores = []
        self.chamfer_distances = []
        self.hausdorff_distances = []
    
    def compute_dice_score(self, prediction, ground_truth):
        """
        Compute Dice score between prediction and ground truth.
        
        Args:
            prediction: Predicted segmentation tensor
            ground_truth: Ground truth segmentation tensor
        
        Returns:
            Dice score
        """
        # Reset metric state
        self.dice_metric.reset()
        
        # Compute metric
        self.dice_metric(prediction, ground_truth)
        
        # Get result
        dice_score = self.dice_metric.aggregate()
        
        return dice_score
    
    def compute_mse_score(self, prediction, ground_truth):
        """
        Compute MSE between prediction and ground truth.
        
        Args:
            prediction: Predicted tensor
            ground_truth: Ground truth tensor
        
        Returns:
            MSE score
        """
        # Reset metric state
        self.mse_metric.reset()
        
        # Compute metric
        self.mse_metric(prediction, ground_truth)
        
        # Get result
        mse_score = self.mse_metric.aggregate()
        
        return mse_score
    
    def compute_chamfer_distance(self, mesh_pred, mesh_true):
        """
        Compute Chamfer distance between predicted and ground truth meshes.
        
        Args:
            mesh_pred: Predicted mesh (PyTorch3D Meshes object)
            mesh_true: Ground truth mesh (PyTorch3D Meshes object)
        
        Returns:
            Chamfer distance
        """
        # Sample points from meshes
        pred_points = mesh_pred.verts_padded()
        true_points = mesh_true.verts_padded()
        
        # Compute Chamfer distance
        chamfer_dist, _ = chamfer_distance(
            pred_points, true_points,
            point_reduction="mean",
            batch_reduction="mean"
        )
        
        return chamfer_dist
    
    def compute_hausdorff_distance(self, mesh_pred, mesh_true):
        """
        Compute Hausdorff distance between predicted and ground truth meshes.
        
        Args:
            mesh_pred: Predicted mesh vertices
            mesh_true: Ground truth mesh vertices
        
        Returns:
            Hausdorff distance
        """
        # Convert to numpy if needed
        if torch.is_tensor(mesh_pred):
            pred_verts = mesh_pred.detach().cpu().numpy()
        else:
            pred_verts = mesh_pred
        
        if torch.is_tensor(mesh_true):
            true_verts = mesh_true.detach().cpu().numpy()
        else:
            true_verts = mesh_true
        
        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        
        # Distance from pred to true
        dist_pred_to_true = cdist(pred_verts, true_verts)
        max_dist_pred_to_true = np.min(dist_pred_to_true, axis=1).max()
        
        # Distance from true to pred
        dist_true_to_pred = cdist(true_verts, pred_verts)
        max_dist_true_to_pred = np.min(dist_true_to_pred, axis=1).max()
        
        # Hausdorff distance is the maximum of these
        hausdorff_dist = max(max_dist_pred_to_true, max_dist_true_to_pred)
        
        return hausdorff_dist
    
    def update_tracking(self, dice_score=None, mse_score=None, chamfer_dist=None, hausdorff_dist=None):
        """
        Update tracking arrays with new scores.
        
        Args:
            dice_score: Dice score to add
            mse_score: MSE score to add
            chamfer_dist: Chamfer distance to add
            hausdorff_dist: Hausdorff distance to add
        """
        if dice_score is not None:
            self.dice_scores.append(dice_score.item() if torch.is_tensor(dice_score) else dice_score)
        
        if mse_score is not None:
            self.mse_scores.append(mse_score.item() if torch.is_tensor(mse_score) else mse_score)
        
        if chamfer_dist is not None:
            self.chamfer_distances.append(chamfer_dist.item() if torch.is_tensor(chamfer_dist) else chamfer_dist)
        
        if hausdorff_dist is not None:
            self.hausdorff_distances.append(hausdorff_dist)
    
    def get_summary_statistics(self):
        """
        Get summary statistics for all tracked metrics.
        
        Returns:
            Dictionary containing mean, std, min, max for each metric
        """
        summary = {}
        
        if self.dice_scores:
            dice_array = np.array(self.dice_scores)
            summary['dice'] = {
                'mean': np.mean(dice_array),
                'std': np.std(dice_array),
                'min': np.min(dice_array),
                'max': np.max(dice_array)
            }
        
        if self.mse_scores:
            mse_array = np.array(self.mse_scores)
            summary['mse'] = {
                'mean': np.mean(mse_array),
                'std': np.std(mse_array),
                'min': np.min(mse_array),
                'max': np.max(mse_array)
            }
        
        if self.chamfer_distances:
            chamfer_array = np.array(self.chamfer_distances)
            summary['chamfer'] = {
                'mean': np.mean(chamfer_array),
                'std': np.std(chamfer_array),
                'min': np.min(chamfer_array),
                'max': np.max(chamfer_array)
            }
        
        if self.hausdorff_distances:
            hausdorff_array = np.array(self.hausdorff_distances)
            summary['hausdorff'] = {
                'mean': np.mean(hausdorff_array),
                'std': np.std(hausdorff_array),
                'min': np.min(hausdorff_array),
                'max': np.max(hausdorff_array)
            }
        
        return summary
    
    def print_summary(self):
        """Print summary statistics for all metrics."""
        summary = self.get_summary_statistics()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for metric_name, stats in summary.items():
            print(f"\n{metric_name.upper()} Scores:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print("="*60)
    
    def save_results(self, filepath):
        """
        Save all tracked metrics to a file.
        
        Args:
            filepath: Path to save the results
        """
        results = {
            'dice_scores': self.dice_scores,
            'mse_scores': self.mse_scores,
            'chamfer_distances': self.chamfer_distances,
            'hausdorff_distances': self.hausdorff_distances,
            'summary': self.get_summary_statistics()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")