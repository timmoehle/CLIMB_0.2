
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from nibabel.orientations import io_orientation
import helpers

# ======================================================================================
# Seed finding operations
# ======================================================================================

def _get_si_axis(affine: np.ndarray) -> int:
    try:
        ornt = io_orientation(affine)
        idx = np.where(ornt[:, 0] == 2)[0]
        if idx.size: return int(idx[0])
    except Exception:
        pass
    return 2

def _voxel_count_along_axis(header: nib.Nifti1Header, axis: int, distance_mm: float) -> int:
    zooms = header.get_zooms()[:3]
    vx = float(zooms[axis])
    if vx <= 0 or not np.isfinite(vx):
        vx = float(np.mean([z for z in zooms if z > 0])) if any(z > 0 for z in zooms) else 1.0
    d = int(np.round(distance_mm / vx))
    return max(d, 1)

def _nanpercentile(data: np.ndarray, pct: float) -> float:
    pct = float(np.clip(pct, 0.0, 100.0))
    return float(np.nanpercentile(data, pct))

def _generate_coherent_si_mask(input_data: np.ndarray,
                               preselection_threshold: float,
                               z_search_distance_cm: float,
                               z_coherent_length_cm: float,
                               z_coherent_threshold: float,
                               header: nib.Nifti1Header,
                               si_axis: int) -> np.ndarray:
    candidate_indices = np.argwhere(input_data > z_coherent_threshold)
    num_candidates = int(candidate_indices.shape[0])

    z_search_d = _voxel_count_along_axis(header, si_axis, z_search_distance_cm * 10.0)
    z_len_vox  = _voxel_count_along_axis(header, si_axis, z_coherent_length_cm * 10.0)

    out_mask = np.zeros_like(input_data, dtype=np.int8)
    dim_sizes = input_data.shape

    step = max(num_candidates // 10, 1) if num_candidates else 1
    for i, idx in enumerate(candidate_indices, start=1):
        cand_val = input_data[tuple(idx)]
        if not (cand_val >= z_coherent_threshold):
            continue

        zi = int(idx[si_axis])
        z_start = max(0, zi - z_search_d)
        z_end   = min(dim_sizes[si_axis], zi + z_search_d + 1)

        run = 0
        run_max = 0
        for zz in range(z_start, z_end):
            probe = list(idx); probe[si_axis] = zz
            v = input_data[tuple(probe)]
            if v >= z_coherent_threshold:
                run += 1; run_max = max(run_max, run)
            else:
                run = 0

        if run_max >= z_len_vox:
            out_mask[tuple(idx)] = 1

    return out_mask

def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndi.label(mask, structure=np.ones((3,3,3), dtype=np.int8))
    if n < 1:
        return np.zeros_like(mask, dtype=np.int8)
    counts = np.bincount(labeled.ravel())
    if counts.size <= 1:
        return np.zeros_like(mask, dtype=np.int8)
    label_idx = int(np.argmax(counts[1:]) + 1)
    return (labeled == label_idx).astype(np.int8)

def compute_seed_mask(input_nii: Path,
                                     out_seg_dir: Path,
                                     *,
                                     selection_percentile: float = 80.0,
                                     z_search_distance_cm: float = 2.0,
                                     z_coherent_length_cm: float = 1.0,
                                     keep_largest_component_only: bool = True,
                                     out_name: str = "SSS_seed_mask.nii.gz",
                                     skip_existing_files: bool = False) -> tuple[Path, dict]:
    out_path = out_seg_dir / out_name
    if skip_existing_files and out_path.exists():
        # If skipping, we still need to return valid paths and dummy metrics
        return out_path, {"seed_voxels": "N/A"}

    img = helpers.load_nii(input_nii)
    data = img.get_fdata().astype(float, copy=True)
    affine, header = img.affine, img.header

    data[data == 0] = np.nan
    if np.all(np.isnan(data)):
        raise ValueError("All voxels are zero/NaN after zero removal.")

    sel_thr = _nanpercentile(data, selection_percentile)

    si_axis = _get_si_axis(affine)

    proto_mask = _generate_coherent_si_mask(
        input_data=data,
        preselection_threshold=sel_thr,          # not used internally, but required by signature
        z_search_distance_cm=z_search_distance_cm,
        z_coherent_length_cm=z_coherent_length_cm,
        z_coherent_threshold=sel_thr,
        header=header,
        si_axis=si_axis,
    )

    init_mask = _keep_largest_component(proto_mask) if keep_largest_component_only else proto_mask
    init_mask[np.isnan(data)] = 0
    init_mask = init_mask.astype(np.uint8)

    helpers.save_like(init_mask, img, out_path, dtype=np.uint8)

    metrics = {
        "selection_threshold_value": float(sel_thr),
        "si_axis": int(si_axis),
        "seed_voxels": int(init_mask.sum())
    }
    return out_path, metrics

# ======================================================================================
# Sobel operations
# ======================================================================================

def _convolve_separable_3d(vol: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
    out = ndi.convolve1d(vol, kx, axis=0, mode="reflect")
    out = ndi.convolve1d(out, ky, axis=1, mode="reflect")
    out = ndi.convolve1d(out, kz, axis=2, mode="reflect")
    return out

def _sobel3d_gradients(vol: np.ndarray, *, spacing: tuple[float, float, float] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vol = vol.astype(np.float32, copy=False)
    D = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    S = np.array([ 1.0, 2.0, 1.0], dtype=np.float32)
    Gx = _convolve_separable_3d(vol, D, S, S)
    Gy = _convolve_separable_3d(vol, S, D, S)
    Gz = _convolve_separable_3d(vol, S, S, D)

    if spacing is not None:
        dx, dy, dz = spacing
        dx = float(dx) if dx else 1.0
        dy = float(dy) if dy else 1.0
        dz = float(dz) if dz else 1.0
        Gx /= dx
        Gy /= dy
        Gz /= dz

    return Gx.astype(np.float32, copy=False), Gy.astype(np.float32, copy=False), Gz.astype(np.float32, copy=False)

def _sobel3d_magnitude(Gx: np.ndarray, Gy: np.ndarray, Gz: np.ndarray) -> np.ndarray:
    return np.sqrt(Gx * Gx + Gy * Gy + Gz * Gz).astype(np.float32, copy=False)

def compute_sobel(in_nii: Path, seg_dir: Path, *, bin_thresh: float = 0.04, scale_by_spacing: bool = True, skip_existing_files: bool = False) -> tuple[Path, Path]:
    base = in_nii.name
    if base.lower().endswith(".nii.gz"):
        stem = base[:-7]
    elif base.lower().endswith(".nii"):
        stem = base[:-4]
    else:
        stem = base
    sobel_mag_p = seg_dir / f"{stem}_3d_sobel.nii.gz"
    sobel_bin_p = seg_dir / f"{stem}_bin_3d_sobel.nii.gz"

    if skip_existing_files and sobel_mag_p.exists() and sobel_bin_p.exists():
        return sobel_mag_p, sobel_bin_p

    ref = helpers.load_nii(in_nii)
    vol = ref.get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {vol.shape}")

    spacing = None
    if scale_by_spacing:
        zooms = ref.header.get_zooms()
        spacing = tuple(zooms[:3]) if len(zooms) >= 3 else None

    Gx, Gy, Gz = _sobel3d_gradients(vol, spacing=spacing)
    mag = _sobel3d_magnitude(Gx, Gy, Gz)

    m = float(mag.max())
    if m > 0:
        mag /= m

    helpers.save_like(mag.astype(np.float32), ref, sobel_mag_p, dtype=np.float32)

    binm = (mag >= bin_thresh).astype(np.float32)
    helpers.save_like(binm.astype(np.float32), ref, sobel_bin_p, dtype=np.float32)

    return sobel_mag_p, sobel_bin_p

# ======================================================================================
# Grow operations
# ======================================================================================

def _grow_geodesic(proto_seed: np.ndarray, allowed_region: np.ndarray, max_iters: int | None) -> np.ndarray:
    if max_iters is None:
        return ndi.binary_propagation(proto_seed, structure=ndi.generate_binary_structure(3, 3),
                                      mask=allowed_region).astype(bool)

    struct = ndi.generate_binary_structure(3, 3)
    current = proto_seed.astype(bool).copy()
    for _ in range(int(max_iters)):
        dil = ndi.binary_dilation(current, structure=struct)
        nxt = (dil & allowed_region) | proto_seed
        if np.array_equal(nxt, current):
            break
        current = nxt
    return current

def grow_mask(seed_mask_path: Path, bin_sobel_mask_path: Path, out_dir: Path, *, max_iters: int = 50, skip_existing_files: bool = False) -> Path:
    out_path = out_dir / "SSS_pre_mask.nii.gz"
    if skip_existing_files and out_path.exists():
        return out_path
    
    seed_img  = helpers.load_nii(seed_mask_path)
    sobel_img = helpers.load_nii(bin_sobel_mask_path)

    seed  = seed_img.get_fdata(dtype=np.float32)
    sobel = sobel_img.get_fdata(dtype=np.float32)

    if seed.shape != sobel.shape:
        raise ValueError(f"Shape mismatch: seed {seed.shape} vs sobel {sobel.shape}")
    if not np.allclose(seed_img.affine, sobel_img.affine, atol=1e-4, rtol=1e-4):
        raise ValueError("Affine mismatch: seed and sobel are not in the same space.")

    seed_bin  = seed  > 0
    sobel_bin = sobel > 0

    allowed_region = ~sobel_bin
    proto_seed = seed_bin & allowed_region

    grown = _grow_geodesic(proto_seed, allowed_region, max_iters=max_iters)
    sss_mask = grown.astype(np.float32)

    helpers.save_like(sss_mask, seed_img, out_path)
    return out_path

# ======================================================================================
# Post-processing operations
# ======================================================================================

def _get_structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=np.uint8)
    elif connectivity == 8:
        return np.ones((3,3), dtype=np.uint8)
    else:
        raise ValueError("Connectivity must be 4 or 8.")

def _keep_largest_component_per_slice(mask3d: np.ndarray, connectivity: int = 8) -> np.ndarray:
    if mask3d.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask3d.shape}.")
    structure = _get_structure(connectivity)

    x, y, z = mask3d.shape
    out = np.zeros_like(mask3d, dtype=np.uint8)

    for k in range(z):
        sl = mask3d[..., k] != 0
        if not sl.any():
            continue

        labels, nlab = ndi.label(sl, structure=structure)
        if nlab <= 1:
            out[..., k] = sl.astype(np.uint8)
            continue

        sizes = np.bincount(labels.ravel())
        if sizes.size <= 1:
            out[..., k] = sl.astype(np.uint8)
            continue

        largest_label = sizes[1:].argmax() + 1
        out[..., k] = (labels == largest_label).astype(np.uint8)
    return out

def _binary_close(mask: np.ndarray) -> np.ndarray:
    bin_mask = mask > 0
    struct = np.ones((3, 3, 3), dtype=bool)
    closed = ndi.binary_closing(bin_mask, structure=struct, iterations=1)
    return closed.astype(np.float32)

def postprocess_mask(pre_mask_path: Path, out_dir: Path, skip_existing_files: bool = False) -> Path:
    out_path = out_dir / "SSS_mask.nii.gz"
    if skip_existing_files and out_path.exists():
        return out_path

    img = helpers.load_nii(pre_mask_path)
    vol = img.get_fdata().astype(np.float32)
    lc = _keep_largest_component_per_slice((vol != 0).astype(np.uint8), connectivity=8).astype(np.float32)
    closed = _binary_close(lc)
    helpers.save_like(closed, img, out_path, dtype=np.float32)
    return out_path


# ======================================================================================
# SSS cutter operations
# ======================================================================================

def _first_top_slice_index(bex: np.ndarray) -> int:
    """First axial slice containing any brain (bex>0), scanning from the top."""
    D = bex.shape[-1]
    for k in range(D - 1, -1, -1):  # top = highest index
        if np.any(bex[..., k] > 0):
            return k
    raise ValueError("No nonzero slice found in brain-extracted mask (bex).")

def _last_bottom_slice_index(bex: np.ndarray) -> int:
    """Last axial slice containing any brain (bex>0) at the bottom."""
    D = bex.shape[-1]
    for k in range(0, D):  # bottom = lowest index
        if np.any(bex[..., k] > 0):
            return k
    raise ValueError("No nonzero slice found in brain-extracted mask (bex).")

def _cm_to_slices(dist_cm: float, dz_mm: float) -> int:
    """Convert distance (cm) to number of z-slices using dz (mm/voxel)."""
    if dz_mm <= 0:
        raise ValueError(f"Invalid z voxel size (mm): {dz_mm}")
    return int(round((dist_cm * 10.0) / dz_mm))

def _compute_crop_bounds_and_refs(bex_img: nib.Nifti1Image, d_start_cm: float, d_cut_cm: float) -> tuple[int, int, int, int, float]:
    """
    Compute inclusive z-bounds [z_lo, z_hi] and refs:
      - k_top (first nonzero from top)
      - k_bottom (last nonzero at bottom)
      - dz_mm (z voxel size)
    """
    bex = bex_img.get_fdata(dtype=np.float32)
    if bex.ndim != 3:
        raise ValueError(f"Expected 3D bex, got shape {bex.shape}")

    k_top    = _first_top_slice_index(bex)
    k_bottom = _last_bottom_slice_index(bex)

    zooms = bex_img.header.get_zooms()
    if len(zooms) < 3:
        raise ValueError("Header zooms do not contain 3 spatial dimensions.")
    dz_mm = float(zooms[2])

    n_start = _cm_to_slices(d_start_cm, dz_mm)
    n_cut   = _cm_to_slices(d_cut_cm,   dz_mm)

    # Move inferior (-z) from top (high index): decreasing index
    k1 = max(0, k_top - n_start)
    k2 = max(0, k1    - n_cut)
    z_lo, z_hi = (min(k1, k2), max(k1, k2))

    return z_lo, z_hi, k_top, k_bottom, dz_mm

def _percentile_from_top_bottom(k: int, k_top: int, k_bottom: int) -> float:
    """
    Map index k to percentile: k_top -> 1.0, k_bottom -> 100.0.
    """
    span = abs(k_top - k_bottom)
    if span == 0:
        return 100.0
    pos = (k_top - k) / span  # 0 at top, 1 at bottom (since top is high index)
    return 1.0 + 99.0 * float(np.clip(pos, 0.0, 1.0))

def cut_sss_mask(subject_dir: Path, sss_mask_path: Path, d_start_cm: float, d_cut_cm: float, logger, skip_existing_files: bool = False):
    """
    Cuts the SSS mask between two axial planes.

    Args:
        subject_dir: The directory of the subject.
        sss_mask_path: Path to the SSS mask file.
        d_start_cm: The distance from the top of the brain to the start of the cut.
        d_cut_cm: The length of the cut.
        logger: The logger.
        skip_existing_files: If true, skips the operation if the output file already exists.

    Returns:
        A dictionary with information about the cut.
    """
    try:
        _, seg_dir = helpers.find_required_folders(subject_dir)
    except FileNotFoundError as e:
        logger.warning(f"Skipping {subject_dir.name}: {e}")
        return None

    out_path = seg_dir / "sss_mask_cut.nii.gz"
    if skip_existing_files and out_path.exists():
        logger.info(f"Skipping SSS mask cutting for {subject_dir.name}: {out_path.name} already exists.")
        # Return dummy info, as the file already exists
        return dict(
            k_top="N/A", k_bottom="N/A", k1="N/A", k2="N/A",
            pct1="N/A", pct2="N/A", dz_mm="N/A",
            d_start_cm=d_start_cm, d_cut_cm=d_cut_cm,
            out_path=out_path,
        )

    try:
        bex_path = helpers.find_nii_file(seg_dir, ["brainmask_dilated"])
    except FileNotFoundError as e:
        logger.info(f"Could not find required files for {subject_dir.name}: {e}")
        return None

    try:
        bex_img = helpers.load_nii(bex_path)
        sss_img = helpers.load_nii(sss_mask_path)

        bex = bex_img.get_fdata(dtype=np.float32)
        sss = sss_img.get_fdata(dtype=np.float32)

        if bex.shape != sss.shape:
            raise ValueError(f"Shape mismatch: bex {bex.shape} vs sss {sss.shape}")
        if not np.allclose(bex_img.affine, sss_img.affine, atol=1e-4, rtol=1e-4):
            raise ValueError("Affine mismatch: bex and sss are not in the same space.")

        z_lo, z_hi, k_top, k_bottom, dz_mm = _compute_crop_bounds_and_refs(bex_img, d_start_cm, d_cut_cm)

        k1, k2 = z_hi, z_lo
        pct1 = _percentile_from_top_bottom(k1, k_top, k_bottom)
        pct2 = _percentile_from_top_bottom(k2, k_top, k_bottom)

        D = sss.shape[-1]
        z_mask = np.zeros(D, dtype=bool)
        z_mask[z_lo : z_hi + 1] = True
        sss_cut = (sss > 0) * z_mask[None, None, :] # Apply the mask to the SSS data

        helpers.save_like(sss_cut.astype(np.float32), sss_img, out_path)

        info = dict(
            k_top=int(k_top),
            k_bottom=int(k_bottom),
            k1=int(k1),
            k2=int(k2),
            pct1=float(pct1),
            pct2=float(pct2),
            dz_mm=float(dz_mm),
            d_start_cm=float(d_start_cm),
            d_cut_cm=float(d_cut_cm),
            out_path=out_path,
        )

        logger.info(
            f"[OK] {subject_dir.name}: "
            f"k_top={k_top}, k_bottom={k_bottom}, k1={k1} ({pct1:.2f}%), k2={k2} ({pct2:.2f}%), "
            f"saved -> {out_path}"
        )
        return info

    except Exception as e:
        logger.error(f"Could not cut SSS mask for {subject_dir.name}: {e}")
        return None
