"""
VABY_MODELS_CVR: VABY forward models for CVR

Forward model for CVR measurement using BOLD MRI and PETCo2

Based on Matlab code written by Joana Pinto, December 2020, Oxford
and adapted from Daniel Bulte 2018 script

Python conversion by Martin Craig 2021, Nottingham

(c) 2021 University of Nottingham
"""
import tensorflow as tf

import numpy as np
#import tensorflow_probability as tfp

from vaby.model import Model, ModelOption
from vaby.utils import ValueList
from vaby.parameter import get_parameter

from ._version import __version__

class CvrPetCo2Model(Model):
    """
    Inference forward model for CVR measurement using PETCo2
    """

    OPTIONS = [
        # Regressor, e.g. physiological data file containing PETCO2 measurements
        ModelOption("regressors", "Regression data (e.g. PETCO2 or O2 time series)", type=str, default=None),
        ModelOption("regressor_types", "Regressor types - comma separated one for each regressor. Supported types: co2, custom", type=str, default="co2"),
        ModelOption("samp_rates", "Regressor sampling rates", unit="Hz", type=ValueList, default=[100,]),

        # Protocol parameters
        ModelOption("baseline", "Length of initial baseline block", unit="s", type=int, default=60),
        ModelOption("data_start_time", "Start of MR data relative to start of regressor data - if not provided will be estimated", unit="s", type=float, default=None),
        ModelOption("tr", "Time between MR volumes", unit="s", type=float, default=None),
        ModelOption("air_pressure", "Barometric pressure", unit="mbar", type=int, default=1020),

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

        if self.regressors is None:
            raise ValueError("A regressor must be provided")

        if isinstance(self.regressors, str):
            self.regressors = np.squeeze(np.loadtxt(self.regressors))

        if self.regressors.ndim < 2:
            self.regressors = self.regressors[..., np.newaxis]
        if self.regressors.ndim != 2:
            raise ValueError("Regressor must be 1D or 2D")

        self.n_regressors = self.regressors.shape[1]
        self.regressor_types = [s.strip() for s in self.regressor_types.split(",")]
        if len(self.regressor_types) != self.n_regressors:
            raise ValueError("Number of regressors provided (%i) does not match number of regressor types (%i)" % (self.n_regressors, len(self.regressor_types)))
        if len(self.samp_rates) != self.n_regressors:
            raise ValueError("Number of regressors provided (%i) does not match number of sampling rates (%i)" % (self.n_regressors, len(self.samp_rates)))

        self.params = []
        regressors = []
        print(self.regressors.shape)
        for idx, regressor_type in enumerate(self.regressor_types):
            if regressor_type == "co2":
                regressors.append(self._preproc_co2(self.regressors[..., idx], self.samp_rates[idx]))
                self.params.append(get_parameter("cvr%i" % (idx+1), mean=1.0, dist="FoldedNormal", prior_var=2000, post_var=10, **options))
            elif regressor_type == "custom":
                raise NotImplementedError()
            else:
                raise ValueError("Unrecognized regressor type: %s" % regressor_type)
        self.regressors = np.array(regressors)

        # Differences between timepoints for quick interpolation. Given a delay
        # time > 0 we can compute co2 = co2[int(delay)] + frac(delay) * diff[int(delay)]
        self.regressor_diffs = np.zeros(self.regressors.shape, dtype=np.float32)
        self.regressor_diffs[:, :-1] = self.regressors[:, 1:] - self.regressors[:, :-1]

        # Min/max values
        self.regressor_mins = np.min(self.regressors, axis=0)
        self.regressor_maxs = np.max(self.regressors, axis=0)

        if self.infer_sig0:
            self.params.append(get_parameter("sig0", mean=1, prior_var=1e9, post_mean=1, post_var=10, post_init=self._init_sig0, **options))
        if self.infer_delay:
            self.params.append(get_parameter("delay", mean=0, prior_var=2500, post_var=10, **options))

    def _init_sig0(self, _param, _t, data):
        return np.mean(data, axis=-1), None

    def fit_glm(self, delay_min=-1, delay_max=1, delay_step=1, progress_cb=None):
        self.log.info("GLM: Doing fitting on %i voxels", self.data_model.n_voxels)
        bold_data = self.data_model.data_flat
        t = self.tpts() # in seconds
        delays = np.arange(delay_min, delay_max+delay_step, delay_step, dtype=np.float32)
        best_resid = np.ones(bold_data.shape[0], dtype=np.float32) * 1e99
        best_delay = np.zeros(bold_data.shape[0], dtype=np.float32)
        best_cvr = np.zeros(bold_data.shape[0], dtype=np.float32)
        best_sig0 = np.zeros(bold_data.shape[0], dtype=np.float32)
        best_modelfit = np.zeros(bold_data.shape, dtype=np.float32)
        for idx, delay in enumerate(delays):
            self.log.info("GLM: fitting with delay=%f", delay)
            delayed_tpts = t - delay
            x = []
            for idx in self.regressors.shape[-1]:
                delayed = np.interp(delayed_tpts, t, self.regressors[..., idx])
                x.append((delayed - self.regressor_mins[idx]) / (self.regressors_maxs[idx] - self.regressor_mins[idx]))

            x.append(np.ones(self.data_model.n_tpts))
            x = np.array(x).T
            for vox in range(bold_data.shape[0]):
                y = bold_data[vox, :]
                beta, resid, _rank, _s = np.linalg.lstsq(x, y)
                model = np.dot(x, beta)
                vox_resid = resid[0]
                if vox_resid < best_resid[vox]:
                    best_delay[vox] = delay
                    # FIXME ignoring anything apart from first beta
                    best_cvr[vox] = beta[0]
                    best_sig0[vox] = beta[-1]
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
        cvr = params[:self.n_regressors]

        extra_param = self.n_regressors
        if self.infer_sig0:
            sig0 = params[extra_param]
            extra_param += 1
        else:
            sig0 = 0

        if self.infer_delay:
            delay = params[extra_param]
            extra_param += 1

            # Apply time delay [W, (S), N] FIXME what is length of regressor
            t_delayed = (tpts - delay) / self.tr
            t_delayed = tf.clip_by_value(t_delayed, 0, len(self.regressors[0])-1)
            t_base = tf.floor(t_delayed)

            # Integer index into the CO2 and diff arrays
            t_base_idx = tf.cast(t_base, tf.int32)

            # Fractional distance to next array index, or 0 if base index was < 0
            t_frac = tf.clip_by_value(t_delayed - t_base, 0, 1)
        else:
            t_base_idx = tf.cast(tf.floor(tpts / self.tr), tf.int32)
            t_frac = None

        fit = 1
        for idx, regressor in enumerate(self.regressors):
            # Tile regressor arrays over all nodes so we can use tf.gather
            regressor = tf.tile(regressor[np.newaxis, ...], (tf.shape(t_base_idx)[0], 1))

            # Grab base and apply linear interpolation
            delayed = tf.gather(regressor, t_base_idx, axis=1, batch_dims=1)

            if t_frac is not None:
                regressor_diff = tf.tile(self.regressor_diffs[idx][np.newaxis, ...], (tf.shape(t_base_idx)[0], 1))
                delayed_diff = tf.gather(regressor_diff, t_base_idx, axis=1, batch_dims=1)
                delayed += t_frac * delayed_diff
            #delayed =  tfp.math.batch_interp_regular_1d_grid(t_delayed, 0, len(self.co2_mmHg), self.co2_mmHg, axis=-1)

            # Sigmoid response
            #return sig0 + (b/(1+c.(e^(-(delayed_co2-c)/d))))/100

            fit += cvr[idx] * (delayed - self.regressor_mins[idx]) / 100

        fit = sig0 * fit
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
        bold_data_average = np.mean(self.data_model.data_flat, axis=0)

        # Interpolate BOLD timeseries onto first regressor FIXME why first?
        mr_timings = self.tpts()
        regressor = self.regressors[..., 0]
        timings = np.array(range(len(regressor)), dtype=np.float32) / self.samp_rates[0]
        bold_data_interp = np.interp(timings, mr_timings, bold_data_average)

        _cc, delay_vols = self._cross_corr(bold_data_interp, regressor)
        delay = delay_vols / self.samp_rates[0] # to seconds
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

    def _preproc_co2(self, co2, samp_rate):
        """
        Preprocess CO2 measurements from physiological data file
        """
        co2 = np.squeeze(co2)

        if self.data_start_time is None:
            # Estimate the time of the first volume in the MR data
            self.data_start_time = max(self.estimate_data_start_time(), 0)

            # Calculate the latest possible start time of the MR data
            # in case the cross correlation method returns something silly
            pco2_duration = len(co2) / samp_rate
            mr_duration = self.tr * self.data_model.n_tpts
            max_time_begin = pco2_duration - mr_duration
            self.data_start_time = min(self.data_start_time, max_time_begin)
            self.log.debug("co2 len: %i", len(co2))
            self.log.debug("mr len: %i", self.data_model.n_tpts)
            self.log.debug("max start: %f", max_time_begin)
        self.log.debug("data start: %f", self.data_start_time)
        co2_trim = co2[int(self.data_start_time * samp_rate):]

        # Determined respiratory frequency during baseline and use info to
        # determine size of end-tidal search window
        baseline_vols = int(self.baseline * samp_rate)
        baseline_fft = np.fft.fft(co2_trim[:baseline_vols])
        p2 = np.abs(baseline_fft/baseline_vols)
        p1 = np.array(p2[:int(baseline_vols/2)+1])
        p1[1:-2] = 2*p1[1:-2]
        f = np.linspace(0, samp_rate/2, int(baseline_vols/2)+1)

        loc = np.argmax(p1[1:])

        pkloc = loc+1
        harm = f[pkloc]
        resp_period = round(1/harm) # e.g. 8s

        # Search window = 1 second more than the respiratory period
        nsearch_vols = int((resp_period+1)*samp_rate)
        windows = int(np.floor(co2_trim.shape[0]/nsearch_vols))

        # Find peak PETCO2 in each window - it's value and index position
        posmax = np.zeros(windows, dtype=np.int)
        winmax = np.zeros(windows)
        for i in range(windows):
            for j in range(nsearch_vols):
                if j == 0 or co2_trim[i*nsearch_vols+j] > winmax[i]:
                    winmax[i] = co2_trim[i*nsearch_vols+j]
                    posmax[i] = i*nsearch_vols+j

        # Make new full sample ET time course where the PETCO2 changes linearly
        # between window maxima
        co2_resamp = np.zeros((co2_trim.shape[0], 1))
        for x in range(windows-1):
            dist_c = posmax[x+1] - posmax[x]
            step_c = winmax[x+1] - winmax[x]
            ramp_c = step_c / dist_c
            for g in range(dist_c+1):
                co2_resamp[posmax[x]+g] = winmax[x] + (ramp_c * g)

        # Pad the start and end with repeats of first and last value to maintain
        # length and phase
        co2_resamp[:posmax[0]] = co2_resamp[posmax[0]]
        co2_resamp[posmax[-1]:] = co2_resamp[posmax[-1]]

        # Create a timecourse of the end tidal CO2 values at the TR's for use with CVR sigmoids
        # Make new time course at the TR resolution and normalise timecourse betwwen 0 and 1 to create EV
        block = int(round(self.tr*samp_rate))
        ev_co2 = np.zeros((self.data_model.n_tpts,), dtype=np.float32)
        for i in range(self.data_model.n_tpts):
            ev_co2[i] = co2_resamp[block * i + block-1]

        # Convert to mmHg
        air_pressure_mmhg = self.air_pressure/1.33322387415 # pressure mbar
        co2_mmHg = (ev_co2 * air_pressure_mmhg) / 100 # div by 100 as values are in percent

        # Calculation of normo/hypercapnea from on/off volumes
        # Convert time periods to number of volumes
        #baseline_vols = self.baseline/self.tr
        #blocksize_on_vols = self.blocksize_on/self.tr
        #blocksize_off_vols = self.blocksize_off/self.tr

        # Average all of first baseline block
        #self.normocap = np.mean(co2_mmHg[:int(baseline_vols+self.delay)])

        #s1 = (baseline_vols+self.delay+blocksize_on_vols/2)
        #s2 = (baseline_vols+self.delay+blocksize_on_vols)
        #s3 = (baseline_vols+self.delay+blocksize_on_vols+blocksize_off_vols+blocksize_on_vols/2)
        #s4 = (baseline_vols+self.delay+blocksize_on_vols+blocksize_off_vols+blocksize_on_vols)
        #s1, s2, s3, s4 = int(s1), int(s2), int(s3), int(s4)
        # Select 2nd half of each hypercapnic block to average
        #hyperblock = np.concatenate([co2_mmHg[s1-1:s2], co2_mmHg[s3-1:s4]])
        #self.hypercap = np.mean(hyperblock)

        return co2_mmHg
