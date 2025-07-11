import os
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from scipy import stats, interpolate

from data.components import UniversalCanonicalResampled, HistogramMatchd


__all__ = ["HistogramUtils", "process_target_datasets", "process_source_datasets", "analyze_contrast_preservation", "analyze_raw_distributions", "analyze_target_cdfs", "analyze_lut_mappings", "sanity_check_target_preservation", "validate_nrrd_support", "main"]


def validate_nrrd_support() -> bool:
    """
    Validate that NRRD support is available.
    
    Returns:
        bool: True if NRRD support is available, False otherwise
    """
    try:
        import nrrd
        return True
    except ImportError:
        print("Warning: pynrrd not installed. NRRD file support disabled.")
        print("Install with: pip install pynrrd")
        return False


class HistogramUtils:
    """
    Shared histogram computation utilities extracted from HistogramMatchd.
    
    These functions handle the core histogram/CDF/LUT computations that are
    used both in offline preprocessing and in the streamlined HistogramMatchd transform.
    """
    
    @staticmethod
    def scale_ct_to_window(img_array: np.ndarray, window_low: float = -200, window_high: float = 800) -> np.ndarray:
        """
        Scale CT image intensities to [0, 1] using clinical windowing.
        
        Uses cardiac CT window [-200, 800] HU to preserve air/bone contrast.
        This prevents over-bright appearance caused by percentile-based scaling.
        
        Args:
            img_array: Input CT image array in Hounsfield Units
            window_low: Lower bound of CT window (default: -200 HU)
            window_high: Upper bound of CT window (default: 800 HU)
        
        Returns:
            Scaled image with intensities in [0, 1] range
        """
        # Clip to window range
        img_clipped = np.clip(img_array, window_low, window_high)
        
        # Scale to [0, 1] range
        img_scaled = (img_clipped - window_low) / (window_high - window_low)
        
        return img_scaled

    @staticmethod
    def scale_to_percentile(img_array: np.ndarray, modal: str = "mr", p_lower: float = 5.0, p_upper: float = 95.0) -> np.ndarray:
        """
        Scale image intensities to [0, 1] with modality-aware approach.
        
        Args:
            img_array: Input image array
            modal: Modality ("ct" or "mr")
            p_lower: Lower percentile for MR (default: 5.0)
            p_upper: Upper percentile for MR (default: 95.0)
        
        Returns:
            Scaled image with intensities in [0, 1] range
        """
        if modal == "ct":
            return HistogramUtils.scale_ct_to_window(img_array)
        else:
            # MR or unknown - use percentile-based scaling
            # Compute percentiles
            low, high = np.percentile(img_array, [p_lower, p_upper])
            
            # Clip to percentile range
            img_clipped = np.clip(img_array, low, high)
            
            # Scale to [0, 1] range
            if high - low > 1e-6:
                img_scaled = (img_clipped - low) / (high - low)
            else:
                img_scaled = np.zeros_like(img_clipped)
            
            return img_scaled

    @staticmethod
    def compute_histogram_raw(img: torch.Tensor, bins: int = 1024) -> np.ndarray:
        """Compute smooth histogram without quantization artifacts."""
        # Convert to numpy and flatten
        if torch.is_tensor(img):
            img_np = img.cpu().numpy()
        else:
            img_np = np.asarray(img)
        
        # Fix uint16 data type handling - PyTorch doesn't support uint16
        if img_np.dtype == np.uint16:
            img_np = img_np.astype(np.float32)
        
        img_flat = img_np.flatten()
        
        # Use consistent 5th-95th percentile scaling
        p_low, p_high = np.percentile(img_flat, [5, 95])
        if p_high - p_low > 1e-6:
            img_scaled = (img_flat - p_low) / (p_high - p_low)
        else:
            img_scaled = np.zeros_like(img_flat)
        
        # Clip to [0,1] range without artificial boundary injection
        img_clipped = np.clip(img_scaled, 0, 1)
        
        # Compute histogram using float64 precision throughout
        # No quantization artifacts - direct histogram computation
        hist, _ = np.histogram(img_clipped, bins=bins, range=(0, 1), density=False)
        
        # Return counts as float64 for precise CDF computation
        return hist.astype(np.float64)

    @staticmethod
    def compute_histogram(img: torch.Tensor, modal: str = "mr", bins: int = 1024, use_raw: bool = True) -> np.ndarray:
        """
        Compute histogram of image intensities. 
        
        Args:
            img: Input image tensor
            modal: Modality (for backwards compatibility)
            bins: Number of histogram bins
            use_raw: If True, use raw distribution-preserving scaling; if False, use modality-specific scaling
        """
        if use_raw:
            return HistogramUtils.compute_histogram_raw(img, bins)
        else:
            # Legacy modality-aware scaling (breaks histogram matching effectiveness)
            # Convert to numpy and flatten
            if torch.is_tensor(img):
                img_np = img.cpu().numpy()
            else:
                img_np = np.asarray(img)
            
            img_flat = img_np.flatten()
            
            # Apply modality-aware scaling
            img_scaled = HistogramUtils.scale_to_percentile(img_flat, modal=modal)
            
            # Ensure full range coverage by adding epsilon values at boundaries
            img_augmented = np.concatenate([
                img_scaled.flatten(),
                [0.0, 1.0]  # Ensure histogram covers full [0,1] range
            ])
            
            # Convert to higher precision range for histogram computation
            img_high_prec = np.clip(img_augmented * (bins - 1), 0, bins - 1).astype(np.uint16)
            
            # Compute histogram
            hist, _ = np.histogram(img_high_prec, bins=bins, range=(0, bins - 1))
            return hist.astype(np.float64)
    
    @staticmethod
    def hist_to_cdf(hist: np.ndarray) -> np.ndarray:
        """Convert histogram to smooth CDF without artificial step patterns."""
        # Normalize to get PDF
        pdf = hist / hist.sum() if hist.sum() > 0 else hist
        
        # Compute CDF using cumulative sum
        cdf = np.cumsum(pdf)
        
        # Ensure proper normalization without forcing artificial boundaries
        if len(cdf) > 0 and cdf[-1] > 0:
            cdf = cdf / cdf[-1]  # Normalize to [0,1]
        
        return cdf
    
    @staticmethod
    def build_intensity_map(src_cdf: np.ndarray, tgt_cdf: np.ndarray, bins: int = 1024) -> np.ndarray:
        """Build smooth intensity mapping with interpolation to eliminate step patterns."""
        from scipy import interpolate
        
        # Initialize lookup table with float64 precision
        lut = np.zeros(bins, dtype=np.float64)
        
        # Create target CDF interpolation function
        # Handle potential duplicate values by using unique values
        tgt_unique_probs, tgt_unique_indices = np.unique(tgt_cdf, return_index=True)
        
        # Ensure we have at least 2 points for interpolation
        if len(tgt_unique_probs) < 2:
            # Fallback to linear mapping if CDF is too flat
            return np.linspace(0, 1, bins, dtype=np.float32)
        
        # Create interpolation function
        # Map probabilities to normalized intensities [0,1]
        tgt_intensities = tgt_unique_indices / (len(tgt_cdf) - 1)
        
        # Create interpolation function with extrapolation
        interp_func = interpolate.interp1d(
            tgt_unique_probs, 
            tgt_intensities,
            kind='linear',
            bounds_error=False,
            fill_value=(0.0, 1.0)  # Extrapolate boundaries
        )
        
        # For each source intensity level, interpolate target intensity
        for i in range(bins):
            src_prob = src_cdf[i]
            
            # Use interpolation to find smooth target intensity
            tgt_intensity = interp_func(src_prob)
            
            # Ensure valid range
            lut[i] = np.clip(tgt_intensity, 0.0, 1.0)
        
        return lut.astype(np.float32)


def discover_dataset_cases(data_root: str, dataset: str, modal: str) -> List[Dict[str, str]]:
    """
    Discover image and label file pairs for a given dataset, assuming a
    Medical Segmentation Decathlon-style 'imagesTr'/'labelsTr' structure.
    
    Args:
        data_root: Root directory containing dataset folders
        dataset: Dataset name (e.g., 'cap', 'mmwhs', 'acdc', 'scotheart')
        modal: Modality ('ct' or 'mr')
    
    Returns:
        List of dictionaries with 'image' and 'label' file paths
    """
    # Map dataset names to actual directory names
    dataset_map = {
        'cap': 'Dataset011_CAP_SAX',
        'mmwhs': 'Dataset022_MMWHS_CT',
        'acdc': 'Dataset021_ACDC',
        'scotheart': 'Dataset020_SCOTHEART'
    }
    dataset_folder = dataset_map.get(dataset.lower(), dataset)
    dataset_dir = Path(data_root) / dataset_folder
    
    image_dir = dataset_dir / "imagesTr"
    label_dir = dataset_dir / "labelsTr"

    if not image_dir.exists() or not label_dir.exists():
        print(f"Warning: 'imagesTr' or 'labelsTr' directory not found in {dataset_dir}. Skipping.")
        return []
    
    cases = []
    # Support both .nii.gz and .nrrd files
    image_files = glob.glob(str(image_dir / "*.nii.gz")) + glob.glob(str(image_dir / "*.nrrd"))
    
    for image_file_str in image_files:
        image_file = Path(image_file_str)
        # Handle different file extensions
        if image_file.suffix == '.nrrd':
            # For .nrrd files, handle .seq.nrrd → .seg.nrrd pattern
            stem = image_file.stem.replace('.seq', '')
            if stem.endswith('_0000'):
                base_name = stem[:-5]
            else:
                base_name = stem
            label_file = label_dir / f"{base_name}.seg.nrrd"
        else:
            # For .nii.gz files, use existing logic
            stem = image_file.stem.replace('.nii', '')
            if stem.endswith('_0000'):
                base_name = stem[:-5]
            else:
                base_name = stem
            label_file = label_dir / f"{base_name}.nii.gz"
        
        if label_file.exists():
            cases.append({
                'image': str(image_file),
                'label': str(label_file)
            })
        else:
            # Fallback for patterns where label might just have _label suffix
            if image_file.suffix == '.nrrd':
                label_file_alt = label_dir / f"{base_name}_label.seg.nrrd"
            else:
                label_file_alt = label_dir / f"{base_name}_label.nii.gz"
            if label_file_alt.exists():
                cases.append({
                    'image': str(image_file),
                    'label': str(label_file_alt)
                })

    print(f"Discovered {len(cases)} cases for {dataset} ({modal})")
    return cases


def process_single_case(case_paths: Dict[str, str], dataset: str, modal: str, 
                       target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0), 
                       include_histogram_matching: bool = False) -> np.ndarray:
    """
    Process a single case to extract its histogram.
    
    Uses the same preprocessing pipeline as the training transforms to ensure
    consistency between offline preprocessing and online inference.
    
    Args:
        case_paths: Dictionary with 'image' and 'label' file paths
        dataset: Dataset name
        modal: Modality
        target_spacing: Target spacing for resampling
        include_histogram_matching: Whether to include histogram matching (should be False for sanity checks)
    
    Returns:
        Histogram of the processed image
    """
    # Use the same preprocessing as training (excluding histogram matching for sanity checks)
    keys = ('image', 'label')
    
    # Create the base preprocessing pipeline
    from monai.transforms import Compose
    transforms = [
        UniversalCanonicalResampled(
            keys=keys,
            dataset=dataset,
            modal=modal,
            target_spacing=target_spacing
        )
    ]
    
    # Only include histogram matching if explicitly requested (not for sanity checks)
    if include_histogram_matching:
        transforms.append(
            HistogramMatchd([keys[0]], modal=modal, dataset=dataset, cdf_dir="./cdf_cache", allow_missing_keys=True)
        )
    
    # Create the composed transform
    loader = Compose(transforms)
    
    # Load and preprocess the case
    data = {
        'image': case_paths['image'],
        'label': case_paths['label'],
        'modal': modal,
        'dataset': dataset
    }
    
    try:
        processed_data = loader(data)
        
        # Get the image array
        img_tensor = processed_data['image']
        if hasattr(img_tensor, 'get_array'):
            img_array = img_tensor.get_array()
        else:
            img_array = img_tensor
        
        # Convert to numpy if needed
        if isinstance(img_array, torch.Tensor):
            img_array = img_array.cpu().numpy()
        
        # Use full volume for more representative histogram computation (raw intensities)
        if len(img_array.shape) == 4:  # [C,H,W,D]
            # Sample every 4th slice to balance representativeness with computation
            sample_volume = img_array[0, :, :, ::4]  # Every 4th slice
            sample_data = sample_volume.flatten()
        elif len(img_array.shape) == 3:  # [H,W,D]
            sample_volume = img_array[:, :, ::4]  # Every 4th slice
            sample_data = sample_volume.flatten()
        else:
            sample_data = img_array.flatten()
        
        # Compute histogram from raw intensities to preserve dataset characteristics
        hist = HistogramUtils.compute_histogram(torch.from_numpy(sample_data), modal=modal, use_raw=True)
        return hist
        
    except Exception as e:
        print(f"Error processing case {case_paths['image']}: {e}")
        return None


def process_target_datasets(data_root: str, cdf_cache_dir: str = "./cdf_cache") -> None:
    """
    Stage A: Process target datasets to build reference CDFs.
    
    Processes all cases in target datasets (CAP for MR, MMWHS for CT),
    accumulates their histograms, and saves reference CDFs.
    
    Args:
        data_root: Root directory containing dataset folders
        cdf_cache_dir: Directory to save CDF cache files
    """
    print("Stage A: Processing target datasets for reference CDFs...")
    
    cdf_cache_path = Path(cdf_cache_dir)
    cdf_cache_path.mkdir(parents=True, exist_ok=True)
    
    # Define target datasets
    target_configs = [
        {'dataset': 'cap', 'modal': 'mr'},
        {'dataset': 'mmwhs', 'modal': 'ct'}
    ]
    
    for config in target_configs:
        dataset = config['dataset']
        modal = config['modal']
        
        print(f"\nProcessing target dataset: {dataset} ({modal})")
        
        # Discover cases
        try:
            cases = discover_dataset_cases(data_root, dataset, modal)
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping {dataset}.")
            continue
        
        if not cases:
            print(f"No cases found for {dataset}. Skipping.")
            continue
        
        # Accumulate histograms from all cases
        accumulated_hist = None
        successful_cases = 0
        
        for case in tqdm(cases, desc=f"Processing {dataset} cases"):
            hist = process_single_case(case, dataset, modal)
            
            if hist is not None:
                if accumulated_hist is None:
                    accumulated_hist = hist.copy()
                else:
                    accumulated_hist += hist
                successful_cases += 1
        
        if accumulated_hist is not None and successful_cases > 0:
            # Average the histogram and convert to CDF
            avg_hist = accumulated_hist / successful_cases
            reference_cdf = HistogramUtils.hist_to_cdf(avg_hist)
            
            # Save reference CDF
            cdf_file = cdf_cache_path / f"{dataset}_reference_cdf.npy"
            np.save(cdf_file, reference_cdf)
            
            print(f"Saved reference CDF for {dataset}: {cdf_file}")
            print(f"Successfully processed {successful_cases}/{len(cases)} cases")
        else:
            print(f"Failed to process any cases for {dataset}")


def process_source_datasets(data_root: str, cdf_cache_dir: str = "./cdf_cache") -> None:
    """
    Stage B: Process source datasets to build LUTs.
    
    Processes all cases in source datasets (ACDC for MR, ScoTHeaRT for CT),
    builds LUTs mapping to target datasets, and saves them.
    
    Args:
        data_root: Root directory containing dataset folders
        cdf_cache_dir: Directory to save LUT cache files
    """
    print("\nStage B: Processing source datasets for LUTs...")
    
    cdf_cache_path = Path(cdf_cache_dir)
    
    # Define source->target mappings
    source_configs = [
        {'source': 'acdc', 'target': 'cap', 'modal': 'mr'},
        {'source': 'scotheart', 'target': 'mmwhs', 'modal': 'ct'}
    ]
    
    for config in source_configs:
        source_dataset = config['source']
        target_dataset = config['target']
        modal = config['modal']
        
        print(f"\nProcessing source dataset: {source_dataset} -> {target_dataset} ({modal})")
        
        # Load reference CDF
        reference_cdf_file = cdf_cache_path / f"{target_dataset}_reference_cdf.npy"
        if not reference_cdf_file.exists():
            print(f"Reference CDF not found: {reference_cdf_file}. Run Stage A first.")
            continue
        
        reference_cdf = np.load(reference_cdf_file)
        print(f"Loaded reference CDF from {reference_cdf_file}")
        
        # Discover source cases
        try:
            cases = discover_dataset_cases(data_root, source_dataset, modal)
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping {source_dataset}.")
            continue
        
        if not cases:
            print(f"No cases found for {source_dataset}. Skipping.")
            continue
        
        # Accumulate histograms from source dataset
        accumulated_hist = None
        successful_cases = 0
        
        for case in tqdm(cases, desc=f"Processing {source_dataset} cases"):
            hist = process_single_case(case, source_dataset, modal)
            
            if hist is not None:
                if accumulated_hist is None:
                    accumulated_hist = hist.copy()
                else:
                    accumulated_hist += hist
                successful_cases += 1
        
        if accumulated_hist is not None and successful_cases > 0:
            # Average the histogram and convert to CDF
            avg_hist = accumulated_hist / successful_cases
            source_cdf = HistogramUtils.hist_to_cdf(avg_hist)
            
            # Build LUT mapping source to target with higher precision
            lut = HistogramUtils.build_intensity_map(source_cdf, reference_cdf, bins=1024)
            
            # Save LUT
            lut_file = cdf_cache_path / f"{source_dataset}_to_{target_dataset}_lut.npy"
            np.save(lut_file, lut)
            
            print(f"Saved LUT for {source_dataset} -> {target_dataset}: {lut_file}")
            print(f"Successfully processed {successful_cases}/{len(cases)} cases")
        else:
            print(f"Failed to process any cases for {source_dataset}")


def analyze_raw_distributions(data_root: str, cdf_cache_dir: str = "./cdf_cache") -> None:
    """Analyze raw intensity distributions before any processing."""
    print("=== Raw Distribution Analysis ===")
    
    datasets = [
        {'name': 'mmwhs', 'modal': 'ct', 'type': 'target'},
        {'name': 'scotheart', 'modal': 'ct', 'type': 'source'},
        {'name': 'cap', 'modal': 'mr', 'type': 'target'},
        {'name': 'acdc', 'modal': 'mr', 'type': 'source'}
    ]
    
    for dataset_config in datasets:
        dataset = dataset_config['name']
        modal = dataset_config['modal']
        
        print(f"\nAnalyzing {dataset} ({modal}) - {dataset_config['type']}")
        
        cases = discover_dataset_cases(data_root, dataset, modal)
        if not cases:
            print(f"No cases found for {dataset}")
            continue
        
        # Analyze first few cases to get distribution characteristics
        sample_cases = cases[:min(5, len(cases))]
        all_intensities = []
        
        for case in sample_cases:
            # Load raw data without UniversalCanonicalResampled processing
            try:
                if case['image'].endswith('.nrrd'):
                    # Handle NRRD files using nrrd library
                    if not validate_nrrd_support():
                        print(f"Skipping NRRD file {case['image']}: NRRD support not available")
                        continue
                    import nrrd
                    img_data, header = nrrd.read(case['image'])
                    # Fix uint16 data type handling for NRRD files
                    if img_data.dtype == np.uint16:
                        img_data = img_data.astype(np.float32)
                    all_intensities.extend(img_data.flatten())
                else:
                    # Handle NII files using nibabel
                    import nibabel as nib
                    img = nib.load(case['image'])
                    img_data = img.get_fdata()
                    # Fix uint16 data type handling for NIfTI files
                    if img_data.dtype == np.uint16:
                        img_data = img_data.astype(np.float32)
                    all_intensities.extend(img_data.flatten())
            except ImportError as e:
                print(f"Missing dependency for {case['image']}: {e}")
                continue
            except Exception as e:
                print(f"Error loading {case['image']}: {e}")
                continue
        
        if all_intensities:
            all_intensities = np.array(all_intensities)
            mean_val = np.mean(all_intensities)
            std_val = np.std(all_intensities)
            p5, p95 = np.percentile(all_intensities, [5, 95])
            
            print(f"  Raw intensities - Mean: {mean_val:.1f}, Std: {std_val:.1f}")
            print(f"  5th-95th percentile: [{p5:.1f}, {p95:.1f}]")
            print(f"  Contrast score: {std_val/abs(mean_val) if abs(mean_val) > 1e-6 else 0:.3f}")


def analyze_target_cdfs(cdf_cache_dir: str = "./cdf_cache") -> None:
    """Analyze target dataset CDFs for distribution characteristics."""
    print("\n=== Target CDF Analysis ===")
    
    target_files = [
        ('mmwhs_reference_cdf.npy', 'MMWHS (CT target)'),
        ('cap_reference_cdf.npy', 'CAP (MR target)')
    ]
    
    cache_path = Path(cdf_cache_dir)
    
    for filename, description in target_files:
        cdf_file = cache_path / filename
        if not cdf_file.exists():
            print(f"CDF file not found: {cdf_file}")
            continue
            
        cdf = np.load(cdf_file)
        
        # Analyze CDF characteristics
        # Find where CDF reaches certain percentiles
        p25_idx = np.searchsorted(cdf, 0.25)
        p50_idx = np.searchsorted(cdf, 0.50)
        p75_idx = np.searchsorted(cdf, 0.75)
        
        # Compute distribution spread
        spread = (p75_idx - p25_idx) / len(cdf)
        
        print(f"\n{description}:")
        print(f"  CDF length: {len(cdf)} bins")
        print(f"  25th percentile at bin: {p25_idx} ({p25_idx/len(cdf):.3f})")
        print(f"  50th percentile at bin: {p50_idx} ({p50_idx/len(cdf):.3f})")
        print(f"  75th percentile at bin: {p75_idx} ({p75_idx/len(cdf):.3f})")
        print(f"  Distribution spread: {spread:.3f}")


def analyze_lut_mappings(cdf_cache_dir: str = "./cdf_cache") -> None:
    """Analyze LUT mapping effectiveness and contrast preservation."""
    print("\n=== LUT Mapping Analysis ===")
    
    lut_files = [
        ('scotheart_to_mmwhs_lut.npy', 'mmwhs_reference_cdf.npy', 'ScoTHeaRT → MMWHS (CT)'),
        ('acdc_to_cap_lut.npy', 'cap_reference_cdf.npy', 'ACDC → CAP (MR)')
    ]
    
    cache_path = Path(cdf_cache_dir)
    
    for lut_filename, target_cdf_filename, description in lut_files:
        lut_file = cache_path / lut_filename
        target_cdf_file = cache_path / target_cdf_filename
        
        if not lut_file.exists() or not target_cdf_file.exists():
            print(f"Files not found for {description}")
            continue
            
        lut = np.load(lut_file)
        target_cdf = np.load(target_cdf_file)
        
        # Analyze LUT characteristics
        identity_mapping = np.arange(len(lut), dtype=np.float32) / (len(lut) - 1)
        identity_ratio = np.mean(np.abs(lut - identity_mapping) < 0.01)
        
        # Compute intensity spread preservation
        lut_range = lut.max() - lut.min()
        intensity_spread_preservation = lut_range  # Should be close to 1.0 for good preservation
        
        # Analyze mapping distribution
        lut_std = np.std(lut)
        
        print(f"\n{description}:")
        print(f"  LUT length: {len(lut)} bins")
        print(f"  LUT range: [{lut.min():.3f}, {lut.max():.3f}]")
        print(f"  Identity mapping ratio: {identity_ratio:.1%} (should be < 30%)")
        print(f"  Intensity spread preservation: {intensity_spread_preservation:.3f} (should be > 0.8)")
        print(f"  LUT standard deviation: {lut_std:.3f} (higher = more mapping variation)")
        
        if identity_ratio > 0.3:
            print(f"  ⚠️  WARNING: High identity mapping suggests ineffective histogram matching")
        if intensity_spread_preservation < 0.8:
            print(f"  ⚠️  WARNING: Low intensity spread preservation")
        if lut_std < 0.1:
            print(f"  ⚠️  WARNING: Low LUT variation suggests limited contrast mapping")
        
        # Add detailed LUT quality validation
        validate_lut_quality(str(lut_file), description)


def validate_lut_quality(lut_file: str, description: str, save_debug: bool = False) -> Dict:
    """Enhanced LUT quality validation with detailed debugging output."""
    print(f"\n=== LUT Quality Validation: {description} ===")
    
    if not os.path.exists(lut_file):
        print(f"LUT file not found: {lut_file}")
        return {}
    
    lut = np.load(lut_file)
    
    # Compute comprehensive metrics
    metrics = {}
    
    # 1. Step Pattern Analysis
    diff = np.diff(lut)
    zero_diffs = np.sum(np.abs(diff) < 1e-6)
    metrics['step_ratio'] = zero_diffs / len(diff)
    
    # 2. Unique Value Analysis
    unique_values = len(np.unique(lut))
    metrics['unique_ratio'] = unique_values / len(lut)
    
    # 3. Gradient Analysis
    second_diff = np.diff(diff)
    metrics['gradient_variation'] = np.std(second_diff)
    
    # 4. Clustering Analysis
    hist, bins = np.histogram(lut, bins=50)
    max_cluster = np.max(hist)
    metrics['cluster_ratio'] = max_cluster / len(lut)
    
    # 5. Smoothness Analysis
    large_jumps = np.sum(np.abs(diff) > 0.01)
    metrics['jump_ratio'] = large_jumps / len(diff)
    
    # 6. Dynamic Range Analysis
    metrics['dynamic_range'] = lut.max() - lut.min()
    
    # Print detailed results
    print(f"  Step Pattern Ratio: {metrics['step_ratio']:.3f} (target: <0.3)")
    print(f"  Unique Value Ratio: {metrics['unique_ratio']:.3f} (target: >0.7)")
    print(f"  Gradient Variation: {metrics['gradient_variation']:.6f} (target: <0.01)")
    print(f"  Cluster Ratio: {metrics['cluster_ratio']:.3f} (target: <0.2)")
    print(f"  Jump Ratio: {metrics['jump_ratio']:.3f} (target: <0.1)")
    print(f"  Dynamic Range: {metrics['dynamic_range']:.3f} (target: >0.8)")
    
    # Quality assessment
    quality_score = 0
    if metrics['step_ratio'] < 0.3: quality_score += 1
    if metrics['unique_ratio'] > 0.7: quality_score += 1
    if metrics['gradient_variation'] < 0.01: quality_score += 1
    if metrics['cluster_ratio'] < 0.2: quality_score += 1
    if metrics['jump_ratio'] < 0.1: quality_score += 1
    if metrics['dynamic_range'] > 0.8: quality_score += 1
    
    metrics['quality_score'] = quality_score
    
    print(f"  Overall Quality Score: {quality_score}/6")
    
    if quality_score >= 5:
        print("  ✅ EXCELLENT: LUT shows smooth mapping characteristics")
    elif quality_score >= 3:
        print("  ⚠️  MODERATE: LUT has some smoothness issues")
    else:
        print("  ❌ POOR: LUT shows significant step patterns and clustering")
    
    return metrics


def analyze_contrast_preservation(source_dataset: str, target_dataset: str, modal: str, cdf_cache_dir: str = "./cdf_cache") -> None:
    """Comprehensive contrast preservation analysis for a source→target mapping."""
    print(f"\n=== Contrast Preservation Analysis ===")
    print(f"Dataset: {source_dataset} → {target_dataset} ({modal})")
    
    cache_path = Path(cdf_cache_dir)
    
    # Load LUT and target CDF
    lut_file = cache_path / f"{source_dataset}_to_{target_dataset}_lut.npy"
    target_cdf_file = cache_path / f"{target_dataset}_reference_cdf.npy"
    
    if not lut_file.exists() or not target_cdf_file.exists():
        print(f"Required files not found in {cdf_cache_dir}")
        return
    
    lut = np.load(lut_file)
    target_cdf = np.load(target_cdf_file)
    
    # Comprehensive analysis
    analyze_lut_mappings(cdf_cache_dir)
    
    # Overall effectiveness score
    identity_mapping = np.arange(len(lut), dtype=np.float32) / (len(lut) - 1)
    identity_ratio = np.mean(np.abs(lut - identity_mapping) < 0.01)
    intensity_spread = lut.max() - lut.min()
    contrast_score = np.std(lut)
    
    effectiveness_score = (1 - identity_ratio) * intensity_spread * min(contrast_score * 3, 1.0)
    
    print(f"\nOverall Effectiveness Score: {effectiveness_score:.3f}")
    if effectiveness_score > 0.7:
        print("✅ Histogram matching is effective for this dataset pair")
    elif effectiveness_score > 0.4:
        print("⚠️ Histogram matching shows moderate effectiveness")
    else:
        print("❌ Histogram matching appears ineffective - consider raw intensity approach")


def sanity_check_target_preservation(data_root: str, dataset: str = "mmwhs", modal: str = "ct", 
                                   cdf_cache_dir: str = "./cdf_cache", max_cases: int = 5) -> None:
    """
    Sanity check to validate target dataset histogram preservation.
    
    This function verifies that the histogram matching pipeline preserves target dataset
    distributions by comparing:
    1. Raw target dataset histogram (before UniversalCanonicalResampled)
    2. Processed target dataset histogram (after full preprocessing pipeline)
    
    A correct histogram matching implementation should show identical distributions
    for target datasets since they don't undergo LUT mapping.
    
    Args:
        data_root: Root directory containing dataset folders
        dataset: Target dataset to check (default: "mmwhs")
        modal: Modality (default: "ct")
        cdf_cache_dir: Directory containing cache files
        max_cases: Maximum number of cases to analyze (default: 5)
    """
    print(f"\n=== Target Dataset Sanity Check: {dataset.upper()} ({modal.upper()}) ===")
    print(f"Validating histogram preservation for target dataset...")
    
    # Discover target dataset cases
    cases = discover_dataset_cases(data_root, dataset, modal)
    if not cases:
        print(f"No cases found for {dataset}")
        return
    
    # Limit cases for sanity check
    sample_cases = cases[:min(max_cases, len(cases))]
    print(f"Analyzing {len(sample_cases)} cases from {dataset}")
    
    # Collect histograms from both raw and processed data
    raw_histograms = []
    processed_histograms = []
    
    for i, case in enumerate(tqdm(sample_cases, desc=f"Processing {dataset} cases")):
        try:
            # Method 1: Raw histogram (direct file loading)
            if case['image'].endswith('.nrrd'):
                if not validate_nrrd_support():
                    print(f"Skipping NRRD file {case['image']}: NRRD support not available")
                    continue
                import nrrd
                raw_img_data, _ = nrrd.read(case['image'])
            else:
                import nibabel as nib
                img = nib.load(case['image'])
                raw_img_data = img.get_fdata()
            
            # Sample every 4th slice for consistency with process_single_case
            if len(raw_img_data.shape) == 4:  # [C,H,W,D]
                raw_sample = raw_img_data[0, :, :, ::4].flatten()
            elif len(raw_img_data.shape) == 3:  # [H,W,D]
                raw_sample = raw_img_data[:, :, ::4].flatten()
            else:
                raw_sample = raw_img_data.flatten()
            
            # Convert to float32 to handle different numpy dtypes
            raw_sample_float = raw_sample.astype(np.float32)
            
            # Compute raw histogram using same method as Stage A
            raw_hist = HistogramUtils.compute_histogram(torch.from_numpy(raw_sample_float), modal=modal, use_raw=True)
            raw_histograms.append(raw_hist)
            
            # Method 2: Processed histogram (through UniversalCanonicalResampled pipeline WITHOUT histogram matching)
            # For sanity checks, we do NOT want histogram matching applied to target datasets
            processed_hist = process_single_case(case, dataset, modal, include_histogram_matching=False)
            if processed_hist is not None:
                processed_histograms.append(processed_hist)
            else:
                print(f"Warning: Failed to process case {i+1}")
                
        except Exception as e:
            print(f"Error processing case {case['image']}: {e}")
            continue
    
    if not raw_histograms or not processed_histograms:
        print("❌ Failed to collect sufficient data for comparison")
        return
    
    # Ensure we have same number of histograms for comparison
    min_count = min(len(raw_histograms), len(processed_histograms))
    raw_histograms = raw_histograms[:min_count]
    processed_histograms = processed_histograms[:min_count]
    
    print(f"\nComparing {min_count} histogram pairs...")
    
    # Accumulate and average histograms
    raw_accumulated = np.sum(raw_histograms, axis=0) / len(raw_histograms)
    processed_accumulated = np.sum(processed_histograms, axis=0) / len(processed_histograms)
    
    # Convert to CDFs for comparison
    raw_cdf = HistogramUtils.hist_to_cdf(raw_accumulated)
    processed_cdf = HistogramUtils.hist_to_cdf(processed_accumulated)
    
    # Statistical comparison
    # 1. Kolmogorov-Smirnov test for CDF similarity
    ks_statistic, ks_p_value = stats.ks_2samp(raw_cdf, processed_cdf)
    
    # 2. Mean absolute difference between CDFs
    mean_abs_diff = np.mean(np.abs(raw_cdf - processed_cdf))
    
    # 3. Maximum absolute difference
    max_abs_diff = np.max(np.abs(raw_cdf - processed_cdf))
    
    # 4. Distribution moments comparison
    raw_mean = np.average(np.arange(len(raw_cdf)), weights=raw_accumulated)
    processed_mean = np.average(np.arange(len(processed_cdf)), weights=processed_accumulated)
    
    raw_var = np.average((np.arange(len(raw_cdf)) - raw_mean)**2, weights=raw_accumulated)
    processed_var = np.average((np.arange(len(processed_cdf)) - processed_mean)**2, weights=processed_accumulated)
    
    # Print results
    print(f"\n=== Statistical Comparison Results ===")
    print(f"Kolmogorov-Smirnov Test:")
    print(f"  KS Statistic: {ks_statistic:.6f}")
    print(f"  P-value: {ks_p_value:.6f}")
    
    print(f"\nCDF Differences:")
    print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"  Maximum absolute difference: {max_abs_diff:.6f}")
    
    print(f"\nDistribution Moments:")
    print(f"  Raw mean: {raw_mean:.3f}, Processed mean: {processed_mean:.3f}")
    print(f"  Raw variance: {raw_var:.3f}, Processed variance: {processed_var:.3f}")
    print(f"  Mean difference: {abs(raw_mean - processed_mean):.6f}")
    print(f"  Variance difference: {abs(raw_var - processed_var):.6f}")
    
    # Interpretation and validation
    print(f"\n=== Validation Results ===")
    
    # Define thresholds for acceptable differences
    ks_threshold = 0.05  # KS statistic should be < 0.05 for similar distributions
    mean_diff_threshold = 0.01  # Mean absolute CDF difference should be < 1%
    max_diff_threshold = 0.05  # Maximum CDF difference should be < 5%
    moment_threshold = 5.0  # Moment differences should be < 5 bins
    
    checks = []
    
    # KS test check
    if ks_statistic < ks_threshold:
        checks.append(("✅ KS Test", f"PASS (statistic: {ks_statistic:.6f} < {ks_threshold})"))
    else:
        checks.append(("❌ KS Test", f"FAIL (statistic: {ks_statistic:.6f} >= {ks_threshold})"))
    
    # Mean difference check
    if mean_abs_diff < mean_diff_threshold:
        checks.append(("✅ Mean CDF Difference", f"PASS ({mean_abs_diff:.6f} < {mean_diff_threshold})"))
    else:
        checks.append(("❌ Mean CDF Difference", f"FAIL ({mean_abs_diff:.6f} >= {mean_diff_threshold})"))
    
    # Max difference check
    if max_abs_diff < max_diff_threshold:
        checks.append(("✅ Max CDF Difference", f"PASS ({max_abs_diff:.6f} < {max_diff_threshold})"))
    else:
        checks.append(("❌ Max CDF Difference", f"FAIL ({max_abs_diff:.6f} >= {max_diff_threshold})"))
    
    # Moment checks
    mean_diff = abs(raw_mean - processed_mean)
    var_diff = abs(raw_var - processed_var)
    
    if mean_diff < moment_threshold:
        checks.append(("✅ Mean Preservation", f"PASS (diff: {mean_diff:.3f} < {moment_threshold})"))
    else:
        checks.append(("❌ Mean Preservation", f"FAIL (diff: {mean_diff:.3f} >= {moment_threshold})"))
    
    if var_diff < moment_threshold:
        checks.append(("✅ Variance Preservation", f"PASS (diff: {var_diff:.3f} < {moment_threshold})"))
    else:
        checks.append(("❌ Variance Preservation", f"FAIL (diff: {var_diff:.3f} >= {moment_threshold})"))
    
    # Print all checks
    for check_result, details in checks:
        print(f"{check_result}: {details}")
    
    # Overall assessment
    passed_checks = sum(1 for check, _ in checks if "✅" in check)
    total_checks = len(checks)
    
    print(f"\n=== Overall Assessment ===")
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("🎉 EXCELLENT: Target dataset histogram preservation is working correctly!")
        print("   The preprocessing pipeline maintains identical distributions for target datasets.")
    elif passed_checks >= total_checks * 0.8:
        print("✅ GOOD: Target dataset histogram preservation is mostly working.")
        print("   Minor differences detected but within acceptable range.")
    elif passed_checks >= total_checks * 0.6:
        print("⚠️  MODERATE: Some issues with target dataset histogram preservation.")
        print("   Investigate preprocessing pipeline for potential distribution changes.")
    else:
        print("❌ POOR: Significant target dataset histogram distribution changes detected!")
        print("   The preprocessing pipeline is altering target dataset distributions.")
        print("   This indicates a bug in the histogram matching implementation.")
    
    # Generate comparison plot if matplotlib is available
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot CDFs
        plt.subplot(2, 2, 1)
        bins = np.arange(len(raw_cdf))
        plt.plot(bins, raw_cdf, 'b-', label='Raw CDF', linewidth=2)
        plt.plot(bins, processed_cdf, 'r--', label='Processed CDF', linewidth=2)
        plt.xlabel('Intensity Bin')
        plt.ylabel('Cumulative Probability')
        plt.title(f'{dataset.upper()} CDF Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot histograms (normalized)
        plt.subplot(2, 2, 2)
        raw_hist_norm = raw_accumulated / np.sum(raw_accumulated)
        processed_hist_norm = processed_accumulated / np.sum(processed_accumulated)
        plt.plot(bins, raw_hist_norm, 'b-', label='Raw Histogram', linewidth=2)
        plt.plot(bins, processed_hist_norm, 'r--', label='Processed Histogram', linewidth=2)
        plt.xlabel('Intensity Bin')
        plt.ylabel('Probability Density')
        plt.title(f'{dataset.upper()} Histogram Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot absolute differences
        plt.subplot(2, 2, 3)
        cdf_diff = np.abs(raw_cdf - processed_cdf)
        plt.plot(bins, cdf_diff, 'g-', linewidth=2)
        plt.xlabel('Intensity Bin')
        plt.ylabel('|Raw CDF - Processed CDF|')
        plt.title('Absolute CDF Differences')
        plt.grid(True, alpha=0.3)
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f'KS Statistic: {ks_statistic:.6f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f'Mean Abs Diff: {mean_abs_diff:.6f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f'Max Abs Diff: {max_abs_diff:.6f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f'Mean Diff: {mean_diff:.3f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f'Variance Diff: {var_diff:.3f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, f'Checks Passed: {passed_checks}/{total_checks}', fontsize=14, weight='bold', transform=plt.gca().transAxes)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Summary Statistics')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = Path(cdf_cache_dir) / f"{dataset}_sanity_check.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved: {plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"\nNote: Could not generate comparison plot: {e}")
    
    print(f"\n=== Sanity Check Complete for {dataset.upper()} ===\n")


def main():
    """CLI wrapper for histogram preprocessing."""
    parser = argparse.ArgumentParser(
        description="Offline histogram preprocessing for cross-dataset intensity normalization"
    )
    
    parser.add_argument(
        "data_root",
        help="Root directory containing dataset folders"
    )
    
    parser.add_argument(
        "--cdf-cache-dir",
        default="./cdf_cache",
        help="Directory to save CDF and LUT cache files (default: ./cdf_cache)"
    )
    
    parser.add_argument(
        "--stage",
        choices=["a", "b", "both"],
        default="both",
        help="Which stage to run: 'a' (target CDFs), 'b' (source LUTs), or 'both' (default: both)"
    )
    
    parser.add_argument(
        "--modal",
        choices=["ct", "mr", "both"],
        default="both",
        help="Which modality to process (default: both)"
    )
    
    parser.add_argument(
        "--analyze-raw-distributions",
        action="store_true",
        help="Analyze raw intensity distributions before processing"
    )
    
    parser.add_argument(
        "--analyze-target-cdfs", 
        action="store_true",
        help="Analyze target dataset CDFs"
    )
    
    parser.add_argument(
        "--analyze-lut-mappings",
        action="store_true", 
        help="Analyze LUT mapping effectiveness"
    )
    
    parser.add_argument(
        "--analyze-histograms",
        action="store_true",
        help="Run comprehensive histogram analysis (all analysis functions)"
    )
    
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run only analysis functions without preprocessing stages"
    )
    
    parser.add_argument(
        "--sanity-check-target",
        action="store_true",
        help="Run sanity check for target dataset histogram preservation"
    )
    
    parser.add_argument(
        "--target-dataset",
        choices=["mmwhs", "cap", "scotheart"],
        default="mmwhs",
        help="Target dataset for sanity check (default: mmwhs). Use 'scotheart' to check source CT dataset."
    )
    
    parser.add_argument(
        "--max-cases-sanity",
        type=int,
        default=5,
        help="Maximum number of cases for sanity check (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("=== MorphiNet Histogram Preprocessing ===")
    print(f"Data root: {args.data_root}")
    print(f"Cache directory: {args.cdf_cache_dir}")
    print(f"Stage: {args.stage}")
    print(f"Modality: {args.modal}")
    print()
    
    # Validate data root
    if not os.path.exists(args.data_root):
        print(f"Error: Data root directory does not exist: {args.data_root}")
        return 1
    
    try:
        # Handle analysis commands
        if args.analyze_raw_distributions or args.analyze_histograms or args.analysis_only:
            analyze_raw_distributions(args.data_root, args.cdf_cache_dir)
            
        if args.analyze_target_cdfs or args.analyze_histograms or args.analysis_only:
            analyze_target_cdfs(args.cdf_cache_dir)
            
        if args.analyze_lut_mappings or args.analyze_histograms or args.analysis_only:
            analyze_lut_mappings(args.cdf_cache_dir)
            
        if args.sanity_check_target or args.analysis_only:
            # Determine modality for dataset
            if args.target_dataset in ["mmwhs", "scotheart"]:
                target_modal = "ct"
            else:  # cap
                target_modal = "mr"
            sanity_check_target_preservation(
                args.data_root, 
                args.target_dataset, 
                target_modal, 
                args.cdf_cache_dir, 
                args.max_cases_sanity
            )
            
        # Check if analysis was requested
        analysis_run = any([args.analyze_raw_distributions, args.analyze_target_cdfs, 
                           args.analyze_lut_mappings, args.analyze_histograms, args.analysis_only, 
                           args.sanity_check_target])
        
        # Only run preprocessing if not in analysis-only mode
        if not args.analysis_only and (not analysis_run or args.stage in ["a", "b", "both"]):
            if args.stage in ["a", "both"]:
                process_target_datasets(args.data_root, args.cdf_cache_dir)
            
            if args.stage in ["b", "both"]:
                process_source_datasets(args.data_root, args.cdf_cache_dir)
        
        if not analysis_run and not args.analysis_only:
            print("\n=== Preprocessing completed successfully ===")
        elif analysis_run or args.analysis_only:
            print("\n=== Analysis completed ===")
        return 0
        
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 