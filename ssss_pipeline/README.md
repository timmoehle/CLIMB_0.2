
# SSS Segmentation Pipeline

This pipeline performs segmentation of the superior sagittal sinus (SSS) from MRI data.

## Usage

1.  **Configure the pipeline:** Edit the `config.ini` file to set the desired parameters.
2.  **Run the pipeline:** Execute the `main.py` script from the command line:

    ```
    python main.py
    ```

## Configuration

The `config.ini` file has two sections: `[paths]` and `[parameters]`.

### `[paths]`

*   `parent_dir`: The path to the parent directory containing the subject folders.

### `[parameters]`

*   `subject_whitelist`: A comma-separated list of subject folder names to process. If empty, all subjects will be processed.
*   `default_dilation_mm`: The dilation amount in millimeters for the brain mask.
*   `selection_pct`: The percentile for coherence threshold in seed finding.
*   `finder_z_search_cm`: The search distance along the S/I axis in cm for seed finding.
*   `finder_z_length_cm`: The required coherent run length in cm for seed finding.
*   `finder_keep_largest`: Keep only the largest 3D component in seed finding.
*   `seed_only`: If true, only run the seed finder.
*   `sobel_bin_thresh`: The threshold for the binarized Sobel edge detection.
*   `sobel_scale_by_spacing`: Scale Sobel filter by voxel spacing (recommended if not isotropic).
*   `grow_max_iters`: The maximum number of iterations for the region growing.
*   `d_start_cm`: The distance from the top slice to the first cutting plane in cm.
*   `d_cut_cm`: The additional inferior distance from the first plane to the second cutting plane in cm.
