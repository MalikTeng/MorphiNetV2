"""
Refactored MorphiNet - Backward Compatibility Layer

This file maintains backward compatibility with the original run.py interface
while redirecting to the new modular implementation.

For new code, use run_modular.py directly or import from pipeline.orchestrator.
"""

import warnings
from run_modular import MorphiNetPipeline, create_training_pipeline, create_inference_pipeline

# Issue deprecation warning
warnings.warn(
    "The monolithic run.py has been refactored into modular components. "
    "Consider using 'from run_modular import MorphiNetPipeline' or "
    "'from pipeline.orchestrator import MorphiNetOrchestrator' for new code.",
    DeprecationWarning,
    stacklevel=2
)

# Maintain backward compatibility by aliasing the old class name
TrainPipeline = MorphiNetPipeline

# Re-export factory functions for convenience
__all__ = [
    'TrainPipeline',
    'MorphiNetPipeline', 
    'create_training_pipeline',
    'create_inference_pipeline'
]