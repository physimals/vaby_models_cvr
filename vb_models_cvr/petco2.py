"""
Inference forward model for CVR measurement using PETCo2

Based on Matlab code written by Joana Pinto, December 2020, Oxford
and adapted from Daniel Bulte 2018 script

Python conversion by Martin Craig 2021, Nottingham
"""
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

import numpy as np

from svb.model import Model, ModelOption
from svb.utils import ValueList, TF_DTYPE, NP_DTYPE

from svb.parameter import get_parameter

from svb_models_asl import __version__

class CvrPetCo2Model(Model):
    """
    Inference forward model for CVR measurement using PETCo2
    """

    OPTIONS = [
        # Physiological data file containing PETCO2 measurements with timings
        ModelOption("phys_data", "Physiological data file", type=str, default=None),

        # Protocol parameters
        ModelOption("baseline", "Length of initial baseline block", unit="s", type=int, default=60),
        ModelOption("blocksize_on", "Length of ON block", unit="s", type=int, default=120),
        ModelOption("blocksize_off", "Length of OFF block", unit="s", type=int, default=120),
        ModelOption("samp_rate", "Powerlab sampling rate", unit="Hz", type=int, default=100),
        ModelOption("air_pressure", "Barometric pressure", unit="mbar", type=int, default=1020),
        ModelOption("threshold_trig", "Threshold to detect triggers", type=int, default=3),
        ModelOption("delay", "Mechanical delay", type=int, default=15),

        # Model options
        ModelOption("infer_sig0", "Infer signal offset", type=bool, default=False),
        ModelOption("infer_delay", "Infer delay shift on PETCO2", type=bool, default=False),
        ModelOption("infer_drift", "Infer a linear drift on signal", type=bool, default=False),
        ModelOption("sigmoid_response", "Use sigmoid relationship between PETCO2 and CVR", type=bool, default=False)
    ]

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        self.phys_data = np.loadtxt(self.phys_data)
        self._preproc_co2()

        # Baseline map [W, 1, 1]
        self.dpet_co2 = self.hypercap - self.normocap
        self.baseline = tf.constant(np.mean(self.data_model.data_flattened, axis=1), dtype=TF_DTYPE)

        self.params = [
            get_parameter("mag", mean=1.0, prior_var=1e9, post_var=10),
        ]
        if self.infer_sig0:
            self.params.append(get_parameter("sig0", mean=0, prior_var=1e9, post_var=10))
      
    def evaluate(self, params, tpts):
        """
        FIXME won't work in batch because of timepoints

        :param t: Time values tensor of shape [W, 1, N] or [1, 1, N]
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is [W, S, 1] tensor where W is the number of nodes and
                      S the number of samples. This
                      may be supplied as a [P, W, S, 1] tensor where P is the number of
                      parameters.

        :return: [W, S, N] tensor containing model output at the specified time values
                 and for each time value using the specified parameter values
        """
        mag = params[0]
        if self.infer_sig0:
            sig0 = params[1]
        else:
            sig0 = 0

        return sig0 + mag * tf.reshape(self.baseline, tf.shape(mag)) * self.out_co2 * self.dpet_co2 / 100

    def tpts(self):
        """
        Get the full set of timeseries time values

        FIXME return real time values

        :return: Either a Numpy array of shape [N] or a Numpy array of shape
                 [W, N] for nodewise timepoints.
        """
        return np.linspace(0, self.data_model.n_tpts, num=self.data_model.n_tpts, endpoint=False, dtype=NP_DTYPE)

    def __str__(self):
        return "CVR-PETCO2 model: %s" % __version__

    def _preproc_co2(self):
        """
        Preprocess CO2 measurements from physiological data file
        """
        # Physiological data stored in columns of data file
        self.timings = self.phys_data[:, 0]
        self.petco2 = self.phys_data[:, 1]
        self.peto2 = self.phys_data[:, 2]
        self.trig = self.phys_data[:, 3]

        # Determine number of volumes and TR from MR triggers (optional)
        trim_time = self.timings[self.trig > self.threshold_trig]
        trim_time_diff = trim_time[2:-1] - trim_time[1:-2]
        tr = np.mean(trim_time_diff)
        vols = trim_time.shape[0] - 1

        # Temporal shift of end-tidal time courses by mechanical delay
        # (default 15 seconds)
        trim_time_begin = trim_time[1] # 1st trigger
        trim_time_begin_del = trim_time_begin - self.delay # 1st trigger (in sec) - delay
        idx = np.where(self.timings == trim_time_begin_del)[0][0] # find index of 1st trigger (in sec) - delay
        self.petco2_trim = self.petco2[idx:]

        # Determined respiratory frequency during baseline and use info to
        # determine size of end-tidal search window
        samp_period = 1/self.samp_rate
        length = int(self.baseline * self.samp_rate)
        temper = np.fft.fft(self.petco2_trim[:length])
        p2 = np.abs(temper/length)
        p1 = np.array(p2[:int(length/2)+1])
        p1[1:-2] = 2*p1[1:-2]
        f = np.linspace(0, self.samp_rate/2, int(length/2)+1)

        loc = np.argmax(p1[1:])
        pk = p1[loc+1]

        pkloc = loc+1
        harm = f[pkloc]
        resp_period = round(1/harm); # e.g. 8s

        nsearch = (resp_period+1)*self.samp_rate; # search window = 1 second more than the respiratory period
        windows = int(np.floor(self.petco2_trim.shape[0]/nsearch))

        # Find peaks
        # CO2 trace
        etc = np.zeros((windows, windows))
        k=0
        for i in range(windows):
            localmax = self.petco2_trim[i*nsearch]
            posmax = i*nsearch
            for j in range(nsearch):
                if self.petco2_trim[i*nsearch+j] > localmax:
                    localmax = self.petco2_trim[i*nsearch+j]
                    posmax = i*nsearch+j
            etc[k,0] = posmax
            etc[k,1] = localmax
            i += 1
            k += 1

        # Make new full sample ET time course
        self.petco2_resamp = np.zeros((self.petco2_trim.shape[0], 1))
        for x in range(etc.shape[0]-1):
            dist_c = int(etc[x+1, 0] - etc[x, 0])
            step_c = etc[x+1, 1] - etc[x, 1]
            ramp_c = step_c / dist_c
            for g in range(dist_c+1):
                self.petco2_resamp[int(etc[x, 0])+g] = etc[x, 1] + (ramp_c * g)

        # Pad the start and end with repeats of first and last value to maintain
        # length and phase
        self.petco2_resamp[:int(etc[0, 0])] = self.petco2_resamp[int(etc[0, 0])]
        self.petco2_resamp[int(etc[-1, 0]):] = self.petco2_resamp[int(etc[-1, 0])]

        # Make new time course at the TR resolution, then normalise output
        block = round(tr*self.samp_rate)
        ev_co2 = np.zeros((vols,))
        for i in range(vols):
            ev_co2[i] = self.petco2_resamp[block * i + block-1]

        # Create a timecourse of the end tidal CO2 values at the TR's for use with CVR sigmoids
        co2_tc = np.array(ev_co2)

        # convert to mmHg
        self.air_pressure_mmhg = self.air_pressure/1.33322387415 # pressure mbar
        co2_mmHg = (co2_tc * self.air_pressure_mmhg) / 100 # div by 100 as values are in percent

        # Convert time periods to number of volumes
        self.baseline = self.baseline/tr
        self.blocksize_on = self.blocksize_on/tr
        self.blocksize_off = self.blocksize_off/tr

        # Average all of first baseline block
        self.normocap = np.mean(co2_mmHg[:int(self.baseline+self.delay)])

        s1 = (self.baseline+self.delay+self.blocksize_on/2)
        s2 = (self.baseline+self.delay+self.blocksize_on)
        s3 = (self.baseline+self.delay+self.blocksize_on+self.blocksize_off+self.blocksize_on/2)
        s4 = (self.baseline+self.delay+self.blocksize_on+self.blocksize_off+self.blocksize_on)
        s1, s2, s3, s4 = int(s1), int(s2), int(s3), int(s4)
        # Select 2nd half of each hypercapnic block to average FIXME
        hyperblock = np.concatenate([co2_mmHg[s1-1:s2], co2_mmHg[s3-1:s4]])
        self.hypercap = np.mean(hyperblock)

        # Normalise timecourse betwwen 0 and 1 to create EV
        self.out_co2=(ev_co2-np.min(ev_co2))/(np.max(ev_co2)-np.min(ev_co2))
