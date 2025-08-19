
"""
This module is the main entry point for the SSS segmentation pipeline.
"""
from pathlib import Path
import helpers
import pre_processing
import segmentation
import numpy as np

def main():
    """Main function to run the SSS segmentation pipeline."""
    cfg_candidates = [
        r"C:\Users\timmo\Documents\Boston_projects\01_CLIMB_MS\pys\scripts_0818\ssss_pipeline\config.ini",
        str((Path(__file__).parent / "config.ini").resolve())
    ]

    config = None
    for c in cfg_candidates:
        try:
            cfg = helpers.read_config(c)
            if cfg and cfg.sections():
                config = cfg
                break
        except Exception:
            # Ignore and try next candidate
            pass

    if config is None or not config.sections():
        raise FileNotFoundError("Could not load config.ini from any known location.")

    parent_dir = Path(config.get("paths", "parent_dir", fallback="")).expanduser()
    if not str(parent_dir):
        raise KeyError("Missing 'paths.parent_dir' in config.ini")

    log_dir = parent_dir / "ssss_pipeline_logs"
    logger = helpers.setup_logger(log_dir)
    csv_logger = helpers.setup_csv_logger(log_dir)

    try:
        helpers.validate_parameters(config, logger)
    except ValueError:
        return  # Stop execution if config is invalid

    subject_whitelist = config.get('parameters', 'subject_whitelist', fallback=None)
    if subject_whitelist:
        subject_whitelist = [s.strip() for s in subject_whitelist.split(',')]

    skip_existing_files = config.getboolean('parameters', 'skip_existing_files', fallback=False)

    subjects = helpers.iter_subject_dirs(parent_dir, subject_whitelist)

    logger.info(f"Found {len(subjects)} subject(s) to process.")

    for sub_dir in subjects:
        logger.info(f"Processing subject: {sub_dir.name}")
        error_message = "None"
        seed_voxels = "N/A"
        mask_voxels = "N/A"
        cut_voxels = "N/A"

        try:
            # 1. File preparation
            pre_processing.prepare_directory_structure(sub_dir, logger)
            pre_processing.coregister_subject(sub_dir, logger, skip_existing_files)

            # 2. Brain segmentation, cutout and smoothing
            pre_processing.create_brain_mask(sub_dir, logger, skip_existing_files)
            pre_processing.expand_brain_mask(sub_dir, config.getfloat('parameters', 'default_dilation_mm', fallback=3.0), logger, skip_existing_files)
            pre_processing.cutout_brain_from_coreg(sub_dir, logger, skip_existing_files)
            pre_processing.smooth_images(sub_dir, logger, skip_existing_files)

            # 3. SSS Segmentation
            nii_dir, seg_dir = helpers.find_required_folders(sub_dir)
            t1_coreg_bex_smooth_path = helpers.find_nii_file(seg_dir, ["t1", "uni", "post", "coreg", "bex", "smooth"])
            t1_coreg_path = helpers.find_nii_file(nii_dir, ["t1", "uni", "post", "coreg"], ["bex", "smooth"])

            seed_mask_path, seed_metrics = segmentation.compute_seed_mask(
                input_nii=t1_coreg_bex_smooth_path,
                out_seg_dir=seg_dir,
                selection_percentile=config.getfloat('parameters', 'selection_pct', fallback=99.0),
                z_search_distance_cm=config.getfloat('parameters', 'finder_z_search_cm', fallback=0.5),
                z_coherent_length_cm=config.getfloat('parameters', 'finder_z_length_cm', fallback=2.0),
                keep_largest_component_only=config.getboolean('parameters', 'finder_keep_largest', fallback=True),
                skip_existing_files=skip_existing_files,
            )
            seed_voxels = seed_metrics.get("seed_voxels", "N/A")


            if not config.getboolean('parameters', 'seed_only', fallback=False):
                sobel_mag_path, sobel_bin_path = segmentation.compute_sobel(
                    in_nii=t1_coreg_path,
                    seg_dir=seg_dir,
                    bin_thresh=config.getfloat('parameters', 'sobel_bin_thresh', fallback=0.04),
                    scale_by_spacing=config.getboolean('parameters', 'sobel_scale_by_spacing', fallback=True),
                    skip_existing_files=skip_existing_files,
                )

                pre_mask_path = segmentation.grow_mask(
                    seed_mask_path=seed_mask_path,
                    bin_sobel_mask_path=sobel_bin_path,
                    out_dir=seg_dir,
                    max_iters=config.getint('parameters', 'grow_max_iters', fallback=50),
                    skip_existing_files=skip_existing_files,
                )

                postprocess_mask_path = segmentation.postprocess_mask(
                    pre_mask_path=pre_mask_path,
                    out_dir=seg_dir,
                    skip_existing_files=skip_existing_files,
                )
                
                # Get mask voxels
                sss_mask_img = helpers.load_nii(postprocess_mask_path)
                mask_voxels = np.sum(sss_mask_img.get_fdata() > 0)


                cut_info = segmentation.cut_sss_mask(
                    subject_dir=sub_dir,
                    sss_mask_path=postprocess_mask_path,
                    d_start_cm=config.getfloat('parameters', 'd_start_cm', fallback=2),
                    d_cut_cm=config.getfloat('parameters', 'd_cut_cm', fallback=7),
                    logger=logger,
                    skip_existing_files=skip_existing_files,
                )

                if cut_info and "out_path" in cut_info:
                    sss_mask_cut_img = helpers.load_nii(cut_info["out_path"])
                    cut_voxels = np.sum(sss_mask_cut_img.get_fdata() > 0)


        except Exception as e:
            logger.error(f"Error during SSS segmentation for {sub_dir.name}: {e}")
            error_message = str(e).replace(",", ";") # Replace comma to not break CSV

        finally:
            csv_logger.info(f"{sub_dir.name},{seed_voxels},{mask_voxels},{cut_voxels},{error_message}")
            logger.info(f"Finished processing subject: {sub_dir.name}")

if __name__ == "__main__":
    main()
