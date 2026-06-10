"""Path defaults loaded from config.env for MorphiNet entrypoints.

Environment variables override values in config.env. Values in config.env
are intended to be project-relative by default so a checkout can run without
user-specific absolute paths.
"""

import os
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ENV_PATH = PROJECT_ROOT / "config.env"

_DEFAULTS = {
    "MORPHINET_CT_DATA_DIR": "./dataset/Dataset020_SCOTHEART",
    "MORPHINET_MR_DATA_DIR": "./dataset/Dataset011_CAP_SAX",
    "MORPHINET_ACDC_DATA_DIR": "./dataset/Dataset021_ACDC",
    "MORPHINET_MMWHS_DATA_DIR": "./dataset/Dataset022_MMWHS_CT",
    "MORPHINET_CAP_DATA_DIR": "./dataset/Dataset011_CAP_SAX",
    "MORPHINET_SCOTHEART_DATA_DIR": "./dataset/Dataset020_SCOTHEART",
    "MORPHINET_CKPT_DIR": "./checkpoints",
    "MORPHINET_USE_CKPT": "./pretrained",
    "MORPHINET_OUTPUT_ROOT": "./results",
    "MORPHINET_CAP_ALL_FRAME_DIR": "./dataset/Dataset010_CAP_SAX_ALL_FRAME",
    "MORPHINET_CAP_ALL_FRAME_IMAGES_TS_DIR": "./dataset/Dataset010_CAP_SAX_ALL_FRAME/imagesTs",
    "MORPHINET_CAP_NRRD_SOURCE_DIR": "./dataset/Dataset011_CAP_SAX-all_frames",
}

_ENV_CACHE: Dict[str, str] | None = None


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_config_env() -> Dict[str, str]:
    """Parse a shell-compatible KEY=VALUE config.env file."""
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE

    values: Dict[str, str] = {}
    if CONFIG_ENV_PATH.exists():
        for raw_line in CONFIG_ENV_PATH.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            if not key:
                continue
            values[key] = _strip_quotes(value.split(" #", 1)[0])

    _ENV_CACHE = values
    return values


def get_config_value(name: str, default: str | None = None) -> str:
    """Return an environment override, config.env value, or default."""
    fallback = _DEFAULTS.get(name, default)
    value = os.environ.get(name, _load_config_env().get(name, fallback))
    if value is None:
        return ""
    return os.path.expanduser(os.path.expandvars(value))


def get_path_default(name: str, default: str | None = None) -> str:
    """Return a path default for CLI arguments and internal registries."""
    return get_config_value(name, default)


def get_dataset_registry() -> Dict[str, Dict[str, str]]:
    """Return supported inference datasets with configurable data roots."""
    return {
        "acdc": {
            "modality": "mr",
            "json": "./dataset/dataset_task21_f0.json",
            "data_dir": get_path_default("MORPHINET_ACDC_DATA_DIR"),
        },
        "mmwhs": {
            "modality": "ct",
            "json": "./dataset/dataset_task22_f0.json",
            "data_dir": get_path_default("MORPHINET_MMWHS_DATA_DIR"),
        },
        "cap": {
            "modality": "mr",
            "json": "./dataset/dataset_task11_f0.json",
            "data_dir": get_path_default("MORPHINET_CAP_DATA_DIR"),
        },
        "scotheart": {
            "modality": "ct",
            "json": "./dataset/dataset_task20_f0.json",
            "data_dir": get_path_default("MORPHINET_SCOTHEART_DATA_DIR"),
        },
    }
