# Example fitting of ASL model
#
# This example uses the sample multi-PLD data from the FSL course
import sys

import matplotlib.pyplot as plt
import nibabel as nib

from avb import run

model = "cvr_petco2"
outdir = "cvr_example_vb_out"

options = {
    "phys_data" : "phys_data.txt",
    "infer_sig0" : True,
    "save_mean" : True,
    "save_noise" : True,
    #"save_param_history" : True,
    #"save_free_energy_history" : True,
    "save_runtime" : True,
    "save_free_energy" : True,
    "save_model_fit" : True,
    "save_log" : True,
    "save_input_data" : True,
    "save_var" : True,
    "log_stream" : sys.stdout,
}

runtime, avb = run("filtered_func_data.nii.gz", model, outdir, mask="mask.nii.gz", **options)

# Display a single slice (z=10)
#ftiss_img = nib.load("%s/mean_ftiss.nii.gz" % outdir).get_fdata()
#delttiss_img = nib.load("%s/mean_delttiss.nii.gz" % outdir).get_fdata()
#plt.figure("F")
#plt.imshow(ftiss_img[:, :, 10].squeeze())
#plt.figure("delt")
#plt.imshow(delttiss_img[:, :, 10].squeeze())
#plt.show()
