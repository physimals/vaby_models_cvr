import sys

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from vaby_svb.main import run

model = "cvr_petco2"
outdir = "cvr_example_svb_out"

# Inference options
# Note for complete convergence should probably have epochs=500 but this takes a while
options = {
    "phys_data" : "phys_data.txt",
    "infer_sig0" : True,
    "infer_delay" : True,
    "learning_rate" : 0.01,
    "batch_size" : 6,
    "sample_size" : 10,
    "epochs" : 500,
    "log_stream" : sys.stdout,
    "save_mean" : True,
    "save_var" : True,
    #"save_param_history" : True,
    "save_cost" : True,
    #"save_cost_history" : True,
    "save_model_fit" : True,
    "save_log" : True,
}

runtime, svb, training_history = run("filtered_func_data.nii.gz", model, outdir, mask="small.nii.gz", **options)

