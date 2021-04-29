"""
VABY_MODELS_CVR: VABY forward models for CVR

Forward model for CVR measurement using BOLD MRI and PETCo2

Based on Matlab code written by Joana Pinto, December 2020, Oxford
and adapted from Daniel Bulte 2018 script

Python conversion by Martin Craig 2021, Nottingham

(c) 2021 University of Nottingham
"""
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

import numpy as np
import tensorflow_probability as tfp

from vaby.model import Model, ModelOption
from vaby.utils import ValueList
from vaby.parameter import get_parameter

from ._version import __version__

class CvrPetCo2Model(Model):
    """
    Inference forward model for CVR measurement using PETCo2
    """

    OPTIONS = [
        # Physiological data file containing PETCO2 measurements with timings
        ModelOption("phys_data", "Physiological data (PETCO2 or O2 time series)", type=str, default=None),

        # Protocol parameters
        ModelOption("baseline", "Length of initial baseline block", unit="s", type=int, default=60),
        ModelOption("data_start_time", "Start of MR data relative to start of physiological data - if not provided will be estimated", unit="s", type=float, default=None),
        #ModelOption("blocksize_on", "Length of ON block", unit="s", type=int, default=120),
        #ModelOption("blocksize_off", "Length of OFF block", unit="s", type=int, default=120),
        ModelOption("samp_rate", "Powerlab sampling rate", unit="Hz", type=int, default=100),
        ModelOption("tr", "Time between MR volumes", unit="s", type=float, default=None),
        ModelOption("air_pressure", "Barometric pressure", unit="mbar", type=int, default=1020),
        #ModelOption("threshold_trig", "Threshold to detect triggers", type=int, default=3),
        #ModelOption("delay", "Mechanical delay", type=int, default=15),

        # Model options
        ModelOption("infer_sig0", "Infer signal offset", type=bool, default=False),
        ModelOption("infer_delay", "Infer delay shift on PETCO2", type=bool, default=False),
        ModelOption("infer_drift", "Infer a linear drift on signal", type=bool, default=False),
        ModelOption("sigmoid_response", "Use sigmoid relationship between PETCO2 and CVR", type=bool, default=False)
    ]

    def __str__(self):
        return "CVR-PETCO2 model: %s" % __version__

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)

        if isinstance(self.phys_data, str):
            self.phys_data = np.loadtxt(self.phys_data)
        self._preproc_co2()

        self.params = [
            get_parameter("cvr", mean=1.0, dist="FoldedNormal", prior_var=2000, post_var=10, **options),
        ]
        if self.infer_sig0:
            self.params.append(get_parameter("sig0", mean=1, prior_var=1e9, post_mean=1, post_var=10, post_init=self._init_sig0, **options))
        if self.infer_delay:
            self.params.append(get_parameter("delay", mean=0, prior_var=2500, post_var=10, **options))

    def _init_sig0(self, _param, _t, data):
        return np.mean(data, axis=-1), None

    def fit_glm(self, delay_min=-1, delay_max=1, delay_step=1, progress_cb=None):
        self.log.info("GLM: Doing fitting on %i voxels", self.data_model.n_unmasked_voxels)
        bold_data = self.data_model.data_flattened
        t = self.tpts() # in seconds
        delays = np.arange(delay_min, delay_max+delay_step, delay_step, dtype=np.float32)
        best_resid = np.ones(bold_data.shape[0], dtype=np.float32) * 1e99
        best_delay = np.zeros(bold_data.shape[0], dtype=np.float32)
        best_cvr = np.zeros(bold_data.shape[0], dtype=np.float32)
        best_sig0 = np.zeros(bold_data.shape[0], dtype=np.float32)
        best_modelfit = np.zeros(bold_data.shape, dtype=np.float32)
        for idx, delay in enumerate(delays):
            delayed_tpts = t - delay
            delayed_co2 = np.interp(delayed_tpts, t, self.co2_mmHg)
            x = np.array([
                (delayed_co2 - self.min_co2mmHg) / (self.max_co2mmHg - self.min_co2mmHg),
                np.ones(self.data_model.n_tpts)
            ]).T
            self.log.info("GLM: fitting with delay=%f", delay)
            for vox in range(bold_data.shape[0]):
                y = bold_data[vox, :]
                beta, resid, _rank, _s = np.linalg.lstsq(x, y)
                model = np.dot(x, beta)
                vox_resid = resid[0]
                if vox_resid < best_resid[vox]:
                    best_delay[vox] = delay
                    best_cvr[vox] = beta[0]
                    best_sig0[vox] = beta[1]
                    best_modelfit[vox] = model
                    best_resid[vox] = vox_resid
            if progress_cb is not None:
                progress_cb(float(idx)/float(len(delays)))

        self.log.info("GLM: DONE")
        return best_cvr*100/best_sig0 / (self.max_co2mmHg - self.min_co2mmHg), best_delay, best_sig0, best_modelfit

    def evaluate(self, params, tpts):
        """
        FIXME won't work in SVB batch training because of timepoints

        :param t: Time values tensor of shape [W, 1, N] or [1, 1, N]
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is [W, S, 1] tensor where W is the number of nodes and
                      S the number of samples. This
                      may be supplied as a [P, W, S, 1] tensor where P is the number of
                      parameters.

        :return: [W, S, N] tensor containing model output at the specified time values
                 and for each time value using the specified parameter values
        """
        cvr = params[0]

        extra_param = 1
        if self.infer_sig0:
            sig0 = params[extra_param]
            extra_param += 1
        else:
            sig0 = 0

        if self.infer_delay:
            delay = params[extra_param]
            extra_param += 1

            # Apply time delay [W, (S), N]
            t_delayed = (tpts - delay) / self.tr
            t_delayed = tf.clip_by_value(t_delayed, 0, len(self.co2_mmHg)-1)
            t_base = tf.floor(t_delayed)

            # Integer index into the CO2 and diff arrays
            t_base_idx = tf.cast(t_base, tf.int32)

            # Fractional distance to next array index, or 0 if base index was < 0
            t_frac = tf.clip_by_value(t_delayed - t_base, 0, 1)

            # Tile PETCO2 arrays over all nodes so we can use tf.gather
            co2_mmHg = tf.tile(self.co2_mmHg[np.newaxis, ...], (tf.shape(t_base_idx)[0], 1))
            co2_diff = tf.tile(self.co2_diff[np.newaxis, ...], (tf.shape(t_base_idx)[0], 1))

            # Grab base and apply linear interpolation
            delayed_co2_base = tf.gather(co2_mmHg, t_base_idx, axis=1, batch_dims=1)
            delayed_co2_diff = tf.gather(co2_diff, t_base_idx, axis=1, batch_dims=1)
            delayed_co2 = delayed_co2_base + t_frac * delayed_co2_diff
            #delayed_co2 =  tfp.math.batch_interp_regular_1d_grid(t_delayed, 0, len(self.co2_mmHg), self.co2_mmHg, axis=-1)
        else:
            # No delay but still need to use tf.gather because we might only have
            # a sample of the time points in SVB
            t_base_idx = tf.cast(tf.floor(tpts / self.tr), tf.int32)
            co2_mmHg = tf.tile(self.co2_mmHg[np.newaxis, ...], (tf.shape(t_base_idx)[0], 1))
            delayed_co2 = tf.gather(co2_mmHg, t_base_idx, axis=1, batch_dims=1)

        # Sigmoid response
        #return sig0 + (b/(1+c.(e^(-(delayed_co2-c)/d))))/100

        fit = sig0 * (1 + cvr * (delayed_co2 - self.min_co2mmHg) / 100)
        return fit

    def tpts(self):
        """
        Get the full set of timeseries time values

        :return: Either a Numpy array of shape [N] or a Numpy array of shape
                 [W, N] for nodewise timepoints.
        """
        return np.linspace(0, self.data_model.n_tpts, num=self.data_model.n_tpts, endpoint=False, dtype=np.float32) * self.tr

    def estimate_data_start_time(self):
        # Mean time series
        bold_data_average = np.mean(self.data_model.data_flattened, axis=0)

        # Interpolate BOLD timeseries onto PETCO2 trace
        mr_timings = self.tpts()
        co2_timings = np.array(range(len(self.petco2)), dtype=np.float32) / self.samp_rate
        bold_data_interp = np.interp(co2_timings, mr_timings, bold_data_average)

        _cc, delay_vols = self._cross_corr(bold_data_interp, self.petco2)
        delay = delay_vols / self.samp_rate # to seconds
        return -delay

    def _cross_corr(self, y1, y2):
        """
        Calculates the cross correlation and lags

        :param y1: 1D Numpy array
        :param y2: 1D Numpy array, same length as y1

        :return: Tuple of Maximum correlation, lag in terms of the index
        """
        if len(y1) != len(y2):
            raise ValueError('The lengths of the inputs should be the same.')

        corr = np.correlate(y1 - np.mean(y1), 
                            y2 - np.mean(y2),
                            mode='full')
        lag = corr.argmax() - (len(y1) - 1)
        return np.max(corr), lag

    def _preproc_co2(self):
        """
        Preprocess CO2 measurements from physiological data file
        """
        self.petco2 = np.squeeze(self.phys_data)

        if self.data_start_time is None:
            # Estimate the time of the first volume in the MR data
            self.data_start_time = max(self.estimate_data_start_time(), 0)

            # Calculate the latest possible start time of the MR data
            # in case the cross correlation method returns something silly
            pco2_duration = len(self.petco2) / self.samp_rate
            mr_duration = self.tr * self.data_model.n_tpts
            max_time_begin = pco2_duration - mr_duration
            self.data_start_time = min(self.data_start_time, max_time_begin)
            self.log.debug("co2 len: %i", len(self.petco2))
            self.log.debug("mr len: %i", self.data_model.n_tpts)
            self.log.debug("max start: %f", max_time_begin)
        self.log.debug("data start: %f", self.data_start_time)
        petco2_trim = self.petco2[int(self.data_start_time * self.samp_rate):]

        # Determined respiratory frequency during baseline and use info to
        # determine size of end-tidal search window
        baseline_vols = int(self.baseline * self.samp_rate)
        baseline_fft = np.fft.fft(petco2_trim[:baseline_vols])
        p2 = np.abs(baseline_fft/baseline_vols)
        p1 = np.array(p2[:int(baseline_vols/2)+1])
        p1[1:-2] = 2*p1[1:-2]
        f = np.linspace(0, self.samp_rate/2, int(baseline_vols/2)+1)

        loc = np.argmax(p1[1:])

        pkloc = loc+1
        harm = f[pkloc]
        resp_period = round(1/harm) # e.g. 8s

        # Search window = 1 second more than the respiratory period
        nsearch_vols = int((resp_period+1)*self.samp_rate)
        windows = int(np.floor(petco2_trim.shape[0]/nsearch_vols))

        # Find peak PETCO2 in each window - it's value and index position
        posmax = np.zeros(windows, dtype=np.int)
        winmax = np.zeros(windows)
        for i in range(windows):
            for j in range(nsearch_vols):
                if j == 0 or petco2_trim[i*nsearch_vols+j] > winmax[i]:
                    winmax[i] = petco2_trim[i*nsearch_vols+j]
                    posmax[i] = i*nsearch_vols+j

        # Make new full sample ET time course where the PETCO2 changes linearly
        # between window maxima
        self.petco2_resamp = np.zeros((petco2_trim.shape[0], 1))
        for x in range(windows-1):
            dist_c = posmax[x+1] - posmax[x]
            step_c = winmax[x+1] - winmax[x]
            ramp_c = step_c / dist_c
            for g in range(dist_c+1):
                self.petco2_resamp[posmax[x]+g] = winmax[x] + (ramp_c * g)

        # Pad the start and end with repeats of first and last value to maintain
        # length and phase
        self.petco2_resamp[:posmax[0]] = self.petco2_resamp[posmax[0]]
        self.petco2_resamp[posmax[-1]:] = self.petco2_resamp[posmax[-1]]

        # Create a timecourse of the end tidal CO2 values at the TR's for use with CVR sigmoids
        # Make new time course at the TR resolution and normalise timecourse betwwen 0 and 1 to create EV
        block = int(round(self.tr*self.samp_rate))
        ev_co2 = np.zeros((self.data_model.n_tpts,), dtype=np.float32)
        for i in range(self.data_model.n_tpts):
            ev_co2[i] = self.petco2_resamp[block * i + block-1]

        # Convert to mmHg
        air_pressure_mmhg = self.air_pressure/1.33322387415 # pressure mbar
        self.co2_mmHg = (ev_co2 * air_pressure_mmhg) / 100 # div by 100 as values are in percent

        # Differences between timepoints for quick interpolation. Given a delay
        # time > 0 we can compute co2 = co2[int(delay)] + frac(delay) * diff[int(delay)]
        self.co2_diff = np.zeros(len(self.co2_mmHg), dtype=np.float32)
        self.co2_diff[:-1] = self.co2_mmHg[1:] - self.co2_mmHg[:-1]

        # norm/hypercap measures
        self.min_co2mmHg = np.min(self.co2_mmHg)
        self.max_co2mmHg = np.max(self.co2_mmHg)

        # Calculation of normo/hypercapnea from on/off volumes
        # Convert time periods to number of volumes
        #baseline_vols = self.baseline/self.tr
        #blocksize_on_vols = self.blocksize_on/self.tr
        #blocksize_off_vols = self.blocksize_off/self.tr

        # Average all of first baseline block
        #self.normocap = np.mean(self.co2_mmHg[:int(baseline_vols+self.delay)])

        #s1 = (baseline_vols+self.delay+blocksize_on_vols/2)
        #s2 = (baseline_vols+self.delay+blocksize_on_vols)
        #s3 = (baseline_vols+self.delay+blocksize_on_vols+blocksize_off_vols+blocksize_on_vols/2)
        #s4 = (baseline_vols+self.delay+blocksize_on_vols+blocksize_off_vols+blocksize_on_vols)
        #s1, s2, s3, s4 = int(s1), int(s2), int(s3), int(s4)
        # Select 2nd half of each hypercapnic block to average
        #hyperblock = np.concatenate([self.co2_mmHg[s1-1:s2], self.co2_mmHg[s3-1:s4]])
        #self.hypercap = np.mean(hyperblock)
