# FORCASTER
Source code for the FORCASTER and FORCASTEST algorithm.

Data is available at: https://doi.org/10.11588/data/GIWRQA

Also includes as a reference algorithm a CMA-ES minimizer using the NGI objective function.

The config file contains the function get_proj_paths which generates a list with tuples that contain a name, the path to the dicom files that should be calibrated, the path to dicom files with the prior image, and then a list with the calibration algorithms.
Calibration algorithms have a numerical identifier, here are some:

FORCASTER: 33
FORCASTEST: 62
Rough estimator: 60
Refined Estimate: 61
Refined Estimate and CMA-ES with NGI: 70
Only translational calibration: 4
FORCASTER with NGI objective: 42
FORCASTEST with NGI objective: 63
CMA-ES with NGI: -70
CMA-ES with Feature Point Objective: -72
BFGS with NGI: -58
BFGS with FP Objective: -44

Calibration algorithms with a number >= 0 reside in cal.py, those with negative numbers are in cal_bfgs_both.py

For dicom files that are not acquired with an Artis Zeego System the function read_dicoms in load_data.py might need to be adapted. If the dicom files do not contain a coordinate system for each image the creation of the astra geometry in main.py:358 has to be adapted or uncommented to use the astra geometry created with the angles from the standard dicom fields.
