import configparser
import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Optional

import nibabel as nib
import numpy as np


def read_config(config_path: str = 'config.ini'):
    """Reads the configuration file and returns a config object."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# ======================================================================================
# Logging
# ======================================================================================

def setup_logger(log_dir: Path, name: str = "ssss_pipeline") -> logging.Logger:
    """
    Sets up a logger that writes to a file and to the console.

    Args:
        log_dir: The directory to save the log file in.
        name: The name of the logger.

    Returns:
        The configured logger.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(log_dir / f"{name}.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Logger initialized")
    return logger


def setup_csv_logger(log_dir: Path, name: str = "ssss_pipeline_summary") -> logging.Logger:
    """
    Sets up a logger that writes to a CSV file.

    Args:
        log_dir: The directory to save the log file in.
        name: The name of the logger.

    Returns:
        The configured logger.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if the logger already has handlers to prevent adding them multiple times
    if not logger.handlers:
        # Use a different formatter for CSV
        # The values will be comma-separated
        fmt = logging.Formatter("%(message)s")
        
        fh = logging.FileHandler(log_dir / f"{name}.csv")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        
        # Write header if the file is new
        if not (log_dir / f"{name}.csv").exists() or (log_dir / f"{name}.csv").stat().st_size == 0:
            logger.info("Subject,seed_voxels,mask_voxels,cut_voxels,errors")
            
    return logger


def validate_parameters(config: configparser.ConfigParser, logger: logging.Logger):
    """Validates critical parameters from the config file."""
    if 'parameters' not in config:
        logger.warning("No [parameters] section in config.ini, using default values.")
        return

    params = config['parameters']
    checks = {
        'default_dilation_mm': (0, None),
        'selection_pct': (0, 100),
        'finder_z_search_cm': (0, None),
        'finder_z_length_cm': (0, None),
        'sobel_bin_thresh': (0, 1),
        'grow_max_iters': (1, None),
        'd_start_cm': (0, None),
        'd_cut_cm': (0, None),
    }

    for key, (min_val, max_val) in checks.items():
        try:
            val = params.getfloat(key)
            if min_val is not None and val < min_val:
                raise ValueError(f"{key} must be >= {min_val}, but is {val}")
            if max_val is not None and val > max_val:
                raise ValueError(f"{key} must be <= {max_val}, but is {val}")
        except (ValueError, configparser.NoOptionError) as e:
            logger.error(f"Invalid or missing config parameter: [parameters].{key}. Details: {e}")
            raise

# ======================================================================================
# NIfTI I/O
# ======================================================================================

def save_like(data3d: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path, dtype=np.float32) -> None:
    """
    Saves a 3D numpy array as a NIfTI file with the same header and affine as a reference image.

    Args:
        data3d: The 3D numpy array to save.
        ref_img: The reference NIfTI image.
        out_path: The path to save the new NIfTI file to.
        dtype: The data type to save the new NIfTI file with.
    """
    data3d = np.asarray(data3d, dtype=dtype)
    img = nib.Nifti1Image(data3d, affine=ref_img.affine, header=ref_img.header.copy())
    sform, scode = ref_img.get_sform(coded=True)
    qform, qcode = ref_img.get_qform(coded=True)
    if scode > 0: img.set_sform(sform, code=scode)
    if qcode > 0: img.set_qform(qform, code=qcode)
    img.set_data_dtype(dtype)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))

def load_nii(path: Path) -> nib.Nifti1Image:
    """
    Loads a NIfTI file.

    Args:
        path: The path to the NIfTI file.

    Returns:
        The loaded NIfTI image.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    return nib.load(str(path))

# ======================================================================================
# Path operations
# ======================================================================================

def is_nii(p: Path) -> bool:
    """
    Checks if a path is a NIfTI file.

    Args:
        p: The path to check.

    Returns:
        True if the path is a NIfTI file, False otherwise.
    """
    if not p.is_file():
        return False
    if p.suffix == ".nii":
        return True
    return len(p.suffixes) >= 2 and p.suffixes[-2:] == [".nii", ".gz"]

def iter_subject_dirs(parent: Path, whitelist: Optional[Iterable[str]]) -> list[Path]:
    """
    Iterates through the subject directories in a parent directory.

    Args:
        parent: The parent directory.
        whitelist: A list of subject names to include. If None, all subjects are included.

    Returns:
        A list of subject directories.
    """
    allow = None
    if whitelist:
        allow = {w.strip() for w in whitelist if w and w.strip()}
    return [p for p in sorted(parent.iterdir()) if p.is_dir() and (allow is None or p.name in allow)]

def find_required_folders(sub_dir: Path) -> tuple[Path, Path]:
    """
    Finds the 'nii' and 'seg' folders in a subject directory.

    Args:
        sub_dir: The subject directory.

    Returns (nii_dir, seg_dir) for a subject directory.
    """
    nii = sub_dir / "nii"
    seg = sub_dir / "seg"
    if not seg.is_dir():
        raise FileNotFoundError(f"'seg' folder missing in {sub_dir}")
    if not nii.is_dir():
        raise FileNotFoundError(f"'nii' folder missing in {sub_dir}")
    return nii, seg

def find_nii_file(directory: Path, required_substrings: list[str], forbidden_substrings: list[str] = None) -> Path:
    """
    Finds a NIfTI file in a directory that contains a given substring.

    Args:
        directory: The directory to search in.
        required_substrings: A list of substrings that the file name must contain.
        forbidden_substrings: A list of substrings that the file name must not contain.

    Returns:
        The path to the NIfTI file.
    """
    cands = []
    for p in directory.rglob("*"):
        if not p.is_file(): continue
        nm = p.name.lower()
        if not (nm.endswith(".nii") or nm.endswith(".nii.gz")): continue
        if all(s in nm for s in required_substrings):
            if forbidden_substrings and any(f in nm for f in forbidden_substrings):
                continue
            cands.append(p)
    if not cands:
        raise FileNotFoundError(f"No NIfTI in {directory} containing all of {required_substrings}")
    cands.sort(key=lambda x: (len(x.name), -x.stat().st_mtime))
    return cands[0]

def case_insensitive_replace(string: str, old: str, new: str, count: int = 1) -> str:
    """Replace 'old' with 'new' ignoring case, up to 'count' occurrences."""
    pat = re.compile(re.escape(old), flags=re.IGNORECASE)
    return pat.sub(new, string, count=count)

def add_suffix_before_extensions(p: Path, suffix: str) -> Path:
    """
    Insert suffix before full extension (handles .nii and .nii.gz).
    """
    if p.suffix == ".nii":
        return p.with_name(f"{p.stem}{suffix}{p.suffix}")
    if len(p.suffixes) >= 2 and p.suffixes[-2:] == [".nii", ".gz"]:
        base = p.name[:-len(".nii.gz")]
        return p.with_name(f"{base}{suffix}.nii.gz")
    return p.with_name(f"{p.stem}{suffix}{''.join(p.suffixes)}")
