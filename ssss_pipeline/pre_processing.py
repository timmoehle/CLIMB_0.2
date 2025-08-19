"""
This module contains functions for pre-processing of MRI data before SSS segmentation.
"""
from pathlib import Path
import shutil
import subprocess
import filecmp
import numpy as np
from scipy.ndimage import binary_dilation, label, zoom, convolve
import SimpleITK as sitk
import helpers
import nibabel as nib
from scipy.ndimage import gaussian_filter


# ======================= Directory Preparation =======================

def prepare_directory_structure(subject_dir: Path, logger):
    """
    Creates the 'seg' directory if it doesn't exist, and copies the
    T1 pre-contrast UNI image from 'nii' to 'seg'.
    """
    nii_dir = subject_dir / "nii"
    seg_dir = subject_dir / "seg"

    if not nii_dir.is_dir():
        logger.warning(f"Skipping {subject_dir.name}: 'nii' directory not found.")
        return

    # Create seg dir
    seg_dir.mkdir(exist_ok=True)
    logger.info(f"Ensured 'seg' directory exists for {subject_dir.name}")

    try:
        # Find and copy the required file
        t1_pre_uni_path = helpers.find_nii_file(nii_dir, ["t1", "pre", "uni"])
        destination_path = seg_dir / t1_pre_uni_path.name

        if not destination_path.exists() or not filecmp.cmp(str(t1_pre_uni_path), str(destination_path)):
            shutil.copy2(t1_pre_uni_path, destination_path)
            logger.info(f"Copied {t1_pre_uni_path.name} to {seg_dir.name}")
        else:
            logger.info(f"File {destination_path.name} already exists and is identical in {seg_dir.name}.")

    except FileNotFoundError:
        logger.info(f"No T1 pre UNI file found to copy for {subject_dir.name}")
    except Exception as e:
        logger.error(f"Could not copy file for {subject_dir.name}: {e}")


# ======================= Registration (rigid) =======================

def estimate_rigid_transform(moving: sitk.Image, fixed: sitk.Image) -> sitk.Transform:
    """
    Estimate rigid (6 DOF) transform aligning moving->fixed.
    Casts to Float32 for metric evaluation; intensities are not modified in the output.
    """
    fixed_f  = sitk.Cast(fixed,  sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    initial_tx = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.VersorRigid3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300, relaxationFactor=0.5
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(initial_tx, inPlace=False)

    final_tx = R.Execute(fixed_f, moving_f)
    return final_tx

def resample_preserve_intensity(moving: sitk.Image, fixed: sitk.Image, tx: sitk.Transform) -> sitk.Image:
    """
    Resample ORIGINAL moving image into fixed grid with NEAREST-NEIGHBOR to preserve values.
    """
    return sitk.Resample(
        moving,
        fixed,
        tx,
        sitk.sitkNearestNeighbor,
        0,
        moving.GetPixelID()
    )


def coregister_subject(subject_dir: Path, logger, skip_existing_files: bool = False) -> int:
    """
    Process one subject directory for co-registration.
    """
    try:
        nii_dir, _ = helpers.find_required_folders(subject_dir)
    except FileNotFoundError as e:
        logger.warning(f"Skipping co-registration for {subject_dir.name}: {e}")
        return 0

    try:
        ref_path = helpers.find_nii_file(nii_dir, ["t1", "pre", "uni"])
        mov_path = helpers.find_nii_file(nii_dir, ["t1", "post", "uni"])
    except FileNotFoundError as e:
        logger.info(f"Skipping co-registration for {subject_dir.name}: required files not found. {e}")
        return 0

    out_path = helpers.add_suffix_before_extensions(mov_path, "_coreg")
    if skip_existing_files and out_path.exists():
        logger.info(f"Skipping co-registration for {subject_dir.name}: {out_path.name} already exists.")
        return 1 # Return 1 to indicate success, as the file already exists

    logger.info(f"Running Co-registration for Subject: {subject_dir.name}")
    logger.info(f"    Fixed (ref):   {ref_path.name}")
    logger.info(f"    Moving (post): {mov_path.name}")
    logger.info(f"    Output:        {out_path.name}")

    try:
        fixed_img  = sitk.ReadImage(str(ref_path))
        moving_img = sitk.ReadImage(str(mov_path))
        tx = estimate_rigid_transform(moving_img, fixed_img)
        write_img = resample_preserve_intensity(moving_img, fixed_img, tx)
        sitk.WriteImage(write_img, str(out_path), useCompression=True)
        return 1
    except Exception as e:
        logger.error(f"Co-registration failed for {subject_dir.name}: {e}")
        return 0


# ======================= Brain Extraction & Masking =======================

def create_brain_mask(subject_dir: Path, logger, skip_existing_files: bool = False):
    """
    Runs the bse command on the T1 pre UNI image to create a brain mask.
    """
    try:
        _, seg_dir = helpers.find_required_folders(subject_dir)
    except FileNotFoundError as e:
        logger.warning(f"Skipping brain mask creation for {subject_dir.name}: {e}")
        return

    try:
        t1_pre_uni_path = helpers.find_nii_file(seg_dir, ["t1", "pre", "uni"])
    except FileNotFoundError:
        logger.info(f"No T1 pre UNI file found in seg dir for brain mask creation on {subject_dir.name}")
        return

    base_name_no_ext = helpers.add_suffix_before_extensions(t1_pre_uni_path, "").stem
    brain_only_path = seg_dir / f"{base_name_no_ext}_bex.nii.gz"
    brainmask_path = seg_dir / f"{base_name_no_ext}_brainmask.nii.gz"

    if skip_existing_files and brain_only_path.exists() and brainmask_path.exists():
        logger.info(f"Skipping brain mask creation for {subject_dir.name}: {brain_only_path.name} and {brainmask_path.name} already exist.")
        return

    command = ['bse', '-i', str(t1_pre_uni_path), '-o', str(brain_only_path), '--auto', '--mask', str(brainmask_path), '--norotate', '--timer']

    logger.info(f"Executing bse: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Successfully created brain mask for: {t1_pre_uni_path.name}")
        if result.stdout: logger.info(f"bse stdout: {result.stdout}")
        if result.stderr: logger.warning(f"bse stderr: {result.stderr}")
    except Exception as e:
        logger.error(f"Error running bse command for {t1_pre_uni_path.name}: {e}")


def _voxel_size_mm_from_header(img: nib.Nifti1Image) -> np.ndarray:
    """Return (dx, dy, dz) in mm using the header zooms."""
    zooms = img.header.get_zooms()[:3]
    return np.asarray(zooms, dtype=float)


def _make_spherical_struct(dilation_mm: float, vox_mm: np.ndarray) -> np.ndarray:
    """Build a sphere-like 3D structuring element for the given physical radius."""
    r_vox = np.ceil(dilation_mm / np.maximum(vox_mm, 1e-6)).astype(int)
    rx, ry, rz = map(int, r_vox)
    x = (np.arange(-rx, rx + 1) * vox_mm[0]) ** 2
    y = (np.arange(-ry, ry + 1) * vox_mm[1]) ** 2
    z = (np.arange(-rz, rz + 1) * vox_mm[2]) ** 2
    dist2 = x[:, None, None] + y[None, :, None] + z[None, None, :]
    return (dist2 <= (dilation_mm ** 2))


def expand_brain_mask(subject_dir: Path, dilation_mm: float, logger, skip_existing_files: bool = False):
    """Expands the brain mask for a single subject."""
    try:
        _, seg_dir = helpers.find_required_folders(subject_dir)
        brainmask_path = helpers.find_nii_file(seg_dir, ["brainmask"], forbidden_substrings=["dilated"])
    except FileNotFoundError as e:
        logger.warning(f"Skipping mask expansion for {subject_dir.name}: {e}")
        return

    out_path = helpers.add_suffix_before_extensions(brainmask_path, "_dilated")
    if skip_existing_files and out_path.exists():
        logger.info(f"Skipping mask expansion for {subject_dir.name}: {out_path.name} already exists.")
        return

    try:
        img = helpers.load_nii(brainmask_path)
        mask = (img.get_fdata() > 0)
        
        # Keep only largest connected component
        labels, n = label(mask)
        if n > 0:
            mask = (labels == (np.argmax(np.bincount(labels.ravel()[1:])) + 1))

        struct = _make_spherical_struct(dilation_mm, _voxel_size_mm_from_header(img))
        dilated = binary_dilation(mask, structure=struct).astype(np.uint8)

        helpers.save_like(dilated, img, out_path, dtype=np.uint8)
        logger.info(f"Saved dilated mask to {out_path.name}")
    except Exception as e:
        logger.error(f"Could not expand mask for {subject_dir.name}: {e}")


def resample_mask_to_image(mask_data: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Resamples a mask to the shape of a target image using nearest-neighbor interpolation."""
    if mask_data.shape == target_shape:
        return mask_data
    zoom_factors = np.array(target_shape) / np.array(mask_data.shape)
    return zoom(mask_data, zoom_factors, order=0, prefilter=False)


def cutout_brain_from_coreg(subject_dir: Path, logger, skip_existing_files: bool = False):
    """Cuts out the brain from the coregistered T1 image using the dilated brain mask."""
    try:
        nii_dir, seg_dir = helpers.find_required_folders(subject_dir)
        mask_path = helpers.find_nii_file(seg_dir, ["brainmask_dilated"])
        input_path = helpers.find_nii_file(nii_dir, ["t1", "post", "coreg"])
    except FileNotFoundError as e:
        logger.warning(f"Skipping brain cutout for {subject_dir.name}: {e}")
        return

    out_path = seg_dir / helpers.add_suffix_before_extensions(input_path, "_bex").name
    if skip_existing_files and out_path.exists():
        logger.info(f"Skipping brain cutout for {subject_dir.name}: {out_path.name} already exists.")
        return

    try:
        img = helpers.load_nii(input_path)
        img_data = img.get_fdata()
        mask_data = helpers.load_nii(mask_path).get_fdata()

        if mask_data.shape != img_data.shape:
            mask_data = resample_mask_to_image(mask_data, img_data.shape)

        cut = img_data * (mask_data > 0)

        helpers.save_like(cut, img, out_path)
        logger.info(f"Saved brain-extracted image to {out_path.name}")
    except Exception as e:
        logger.error(f"Could not cut out brain for {subject_dir.name}: {e}")


# ======================= Smoothing =======================

def _mean_smooth_3d(data: np.ndarray) -> np.ndarray:
    """3x3x3 mean smoothing with nearest border handling."""
    kernel = np.ones((3, 3, 3), dtype=np.float32)
    valid  = np.isfinite(data).astype(np.float32)
    summed = convolve(np.nan_to_num(data, nan=0.0), kernel, mode="nearest")
    count  = convolve(valid, kernel, mode="nearest")
    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed = summed / count
    smoothed[count == 0] = np.nan
    return smoothed


def _smooth_nifti_file(input_path: Path, output_path: Path, logger) -> Path:
    """
    Load NIfTI, apply 3x3x3 mean smoothing, save to 'output_path'.
    Logs full absolute input/output paths for debugging.
    """
    in_abs  = input_path.resolve()
    out_abs = output_path.resolve()
    logger.info(f"  Smoothing (in -> out): {in_abs}  ->  {out_abs}")

    img = nib.load(str(input_path))
    data = img.get_fdata(dtype=np.float32)
    try:
        zooms = tuple(img.header.get_zooms()[:3])
    except Exception:
        zooms = None
    logger.debug(f"    loaded shape={data.shape}, zooms={zooms}, dtype=float32")

    smoothed_data = _mean_smooth_3d(data)
    logger.debug(f"    smoothed: shape={smoothed_data.shape}, dtype={smoothed_data.dtype}")

    helpers.save_like(smoothed_data, img, output_path, dtype=np.float32)
    logger.info(f"  Saved: {out_abs}")
    return output_path


def smooth_images(subject_dir: Path, logger, skip_existing_files: bool = False):
    """
    Applies smoothing to:
      - f1: file found in nii_dir (['t1','uni','post','coreg'], not 'smooth') -> save in seg_dir
      - f2: file found in seg_dir (['t1','uni','post','coreg','bex'], not 'smooth') -> save in seg_dir
    """
    subj_abs = subject_dir.resolve()
    logger.info(f"Smoothing images for {subject_dir.name}  (subject_dir={subj_abs})")

    try:
        nii_dir, seg_dir = helpers.find_required_folders(subject_dir)
    except FileNotFoundError as e:
        logger.warning(f"Skipping smoothing for {subject_dir.name}: {e}")
        return

    nii_abs = nii_dir.resolve()
    seg_abs = seg_dir.resolve()
    logger.info(f"  nii_dir={nii_abs}")
    logger.info(f"  seg_dir={seg_abs}")

    # ensure seg_dir exists
    seg_dir.mkdir(parents=True, exist_ok=True)

    smoothed_count = 0

    # ---------------- f1: from nii_dir -> write into seg_dir ----------------
    try:
        f1 = helpers.find_nii_file(
            nii_dir,
            required_substrings=["t1", "uni", "post", "coreg"],
            forbidden_substrings=["smooth"],
        )
        f1_abs = f1.resolve()
        out1_name = helpers.add_suffix_before_extensions(f1, "_smooth").name
        out1_path = seg_dir / out1_name
        out1_abs  = out1_path.resolve()

        if skip_existing_files and out1_path.exists():
            logger.info(f"Skipping smoothing for {f1.name}: {out1_path.name} already exists.")
        else:
            logger.info(f"  f1 found:   {f1_abs}")
            logger.info(f"  f1 output:  {out1_abs}")

            _smooth_nifti_file(f1, out1_path, logger)
            smoothed_count += 1
    except FileNotFoundError:
        logger.warning(
            f"No suitable f1 in {nii_dir.name} for {subject_dir.name} "
            f"(needs 't1','uni','post','coreg' and not 'smooth')."
        )
    except Exception as e:
        logger.error(f"Failed to smooth f1 ({subject_dir.name}) – input={f1 if 'f1' in locals() else 'N/A'}: {e}")

    # ---------------- f2: from seg_dir -> write into seg_dir ----------------
    try:
        f2 = helpers.find_nii_file(
            seg_dir,
            required_substrings=["t1", "uni", "post", "coreg", "bex"],
            forbidden_substrings=["smooth"],
        )
        f2_abs = f2.resolve()
        out2_name = helpers.add_suffix_before_extensions(f2, "_smooth").name
        out2_path = f2.parent / out2_name
        out2_abs  = out2_path.resolve()

        if skip_existing_files and out2_path.exists():
            logger.info(f"Skipping smoothing for {f2.name}: {out2_path.name} already exists.")
        else:
            logger.info(f"  f2 found:   {f2_abs}")
            logger.info(f"  f2 output:  {out2_abs}")

            _smooth_nifti_file(f2, out2_path, logger)
            smoothed_count += 1
    except FileNotFoundError:
        logger.warning(
            f"No suitable f2 in {seg_dir.name} for {subject_dir.name} "
            f"(needs 't1','uni','post','coreg','bex' and not 'smooth')."
        )
    except Exception as e:
        logger.error(f"Failed to smooth f2 ({subject_dir.name}) – input={f2 if 'f2' in locals() else 'N/A'}: {e}")

    logger.info(f"Finished smoothing for {subject_dir.name}. Smoothed {smoothed_count} file(s).")

