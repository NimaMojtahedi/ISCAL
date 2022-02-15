# this file provides all feature engineering functions

class FeatureEng:

    """
    Feature engineering class contains method for feature engineering.

    INPUT
    data: input data in a form of vector
    fs: sampling frequency
    """

    def __init__(self, data, fs):

        # initializing FeatureEng class
        self.data = data
        self.fs = fs

    def power_spec(self, keep_f):
        """
        power_spec method takes signal as input and calculates power spectral density
        INPUT
        keep_f: keep frequency
        """

        # import libraries
        from scipy.signal import welch

        f, Pxx_den = welch(x=self.data, fs=self.fs,
                           nperseg=self.fs * 2, nfft=self.fs * 4)
        return Pxx_den[f < keep_f]

    def histogram(self, nr_bins=100, range=[-3, 3], as_density=True):
        """
        histogram function calculates histogram of input function for given input data
        INPUT
        nr_bins: resolution of histogram
        range: numeric range to calculate histogram
        as_density: return result as density or count
        """

        # import libraries
        import numpy as np

        # calculate histogram
        my_hist = np.histogram(a=self.data,
                               bins=np.linspace(range[0], range[1], nr_bins+1),
                               density=as_density)[0]

        return my_hist

    def hilbert_transform(self):
        """
        hilbert transofrm on input data
        This function return amplitude envelope and instantaneous phase and instantaneous frequency
        """
        # import necessary libraries
        from scipy.signal import hilbert
        import numpy as np

        # calculate hilbert transformation
        analytic_signal = hilbert(self.data)

        # get amplitude envelope of the signal
        amplitude_envelope = np.abs(analytic_signal)

        # get instantaneous phase
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        # get instantaneous frequency
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                   (2.0*np.pi) * self.fs)

        return amplitude_envelope, instantaneous_frequency, instantaneous_phase

    def autocorrelation(self, lag=1):
        """
        autocorrelation function calculates auto correlation for given signal until lag = n
        """

        # necesary libs
        import numpy as np

        # full correlation
        my_corr = np.correlate(self.signal, self.signal, mode='full')

        return my_corr[int(len(my_corr)/2):]

    def filter(self):
        """
        apply different signal filtering
        """
        pass
