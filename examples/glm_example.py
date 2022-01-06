import sys
import os

from vaby.data import DataModel
from vaby.utils import setup_logging
from vaby_models_cvr import CvrPetCo2Model

import numpy as np
pco2 = np.loadtxt("pco2.txt")

options = {
    "regressors" : pco2,
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

data_model.model_space.save_data(cvr, "cvr_glm", OUTDIR)
data_model.model_space.save_data(delay, "delay_glm", OUTDIR)
data_model.model_space.save_data(sig0, "sig0_glm", OUTDIR)
data_model.model_space.save_data(modelfit, "modelfit_glm", OUTDIR)
