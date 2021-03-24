# Example fitting of ASL model
#
# This example uses the sample multi-PLD data from the FSL course
import sys

import matplotlib.pyplot as plt
import nibabel as nib

from varbay_avb import run

model = "cvr_petco2"
outdir = "cvr_example_vb_out"

options = {
    "phys_data" : "phys_data.txt",
    "infer_sig0" : True,
    "infer_delay" : True,
    "save_mean" : True,
    "save_noise" : True,
    "save_param_history" : True,
    #"save_free_energy_history" : True,
    "save_runtime" : True,
    "save_free_energy" : True,
    "save_model_fit" : True,
    "save_log" : True,
    "save_input_data" : True,
    "save_var" : True,
    "log_stream" : sys.stdout,
    "max_iterations" : 50,
}

runtime, avb = run("filtered_func_data.nii.gz", model, "cvr_example_vb_out", mask="small.nii.gz", **options)
#runtime, avb = run("raw_bold.nii.gz", model, "raw_bold_vb_out", mask="mask.nii.gz", **options)

