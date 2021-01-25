"""
Inference forward model for CVR measurement using PETCo2
"""
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

import numpy as np

from svb.model import Model, ModelOption
from svb.utils import ValueList
from svb.parameter import get_parameter

from svb_models_asl import __version__

class CvrPetCo2Model(Model):
    """
    """

    OPTIONS = [
        ModelOption("t1", "Tissue T1 value", units="s", type=float, default=1.3),
    ]

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)

    def evaluate(self, params, tpts):
        """
        :param t: Time values tensor of shape [W, 1, N] or [1, 1, N]
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is [W, S, 1] tensor where W is the number of nodes and
                      S the number of samples. This
                      may be supplied as a [P, W, S, 1] tensor where P is the number of
                      parameters.

        :return: [W, S, N] tensor containing model output at the specified time values
                 and for each time value using the specified parameter values
        """
        pass

    def __str__(self):
        return "CVR-PETCO2 model: %s" % __version__

