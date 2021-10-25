import sys
import os

from vaby.data import DataModel
from vaby.utils import setup_logging
from vaby_models_cvr import CvrPetCo2Model

options = {
    "phys_data" : "pco2.txt",
    "tr" : 0.8,
    "save_mean" : True,
    #"save_free_energy_history" : True,
    "save_runtime" : True,
    "save_model_fit" : True,
    "save_input_data" : True,
    "save_log" : True,
    "log_stream" : sys.stdout,
    "max_iterations" : 50,
}

OUTDIR="glm_example_out"
os.makedirs(OUTDIR, exist_ok=True)

setup_logging(OUTDIR, **options)
data_model = DataModel("filtered_func_data.nii.gz", mask="mask.nii.gz")
model = CvrPetCo2Model(data_model, **options)
cvr, delay, sig0, modelfit = model.fit_glm(delay_min=-10, delay_max=10, delay_step=2)

cvr_nii = data_model.nifti_image(cvr)
delay_nii = data_model.nifti_image(delay)
sig0_nii = data_model.nifti_image(sig0)
modelfit_nii = data_model.nifti_image(modelfit)
cvr_nii.to_filename(os.path.join(OUTDIR, "cvr_glm.nii.gz"))
delay_nii.to_filename(os.path.join(OUTDIR, "delay_glm.nii.gz"))
sig0_nii.to_filename(os.path.join(OUTDIR, "sig0_glm.nii.gz"))
modelfit_nii.to_filename(os.path.join(OUTDIR, "modelfit_glm.nii.gz"))
