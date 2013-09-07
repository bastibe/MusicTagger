import numpy as np
import scipy.stats
import scipy.signal


def rms(data, fft_data=None):
    """Calculates a measure for the loudness of the data."""
    return np.sqrt(np.mean(data*data))


def peak(data, fft_data=None):
    """Calculated the maximum value in the data."""
    return np.max(np.abs(data))


def crest_factor(data, fft_data=None):
    """Calculates the ratio of peak to rms of the data."""
    peak_val = peak(data)
    rms_val = rms(data)
    if peak_val == 0:
        return 1
    return peak_val/rms_val


def spectral_centroid(data, fft_data):
    """Calculates the balance point of the spectrum."""
    freq = np.fft.fftfreq(len(data))[:len(fft_data)]
    freq[-1] += 1 # last element needs to be 0.5 not -0.5
    spec = np.abs(fft_data)
    spec_sum = np.sum(spec)
    if spec_sum == 0:
        return 0.25
    centroid = np.sum(spec*freq)/spec_sum
    return centroid


def log_spectral_centroid(data, fft_data):
    """Calculates the logarithmic balance point of the spectrum."""
    freq = np.fft.fftfreq(len(data))[:len(fft_data)]
    freq[-1] += 1 # last element needs to be 0.5 not -0.5
    freq_log = np.log(freq+1) # linear range into 1 to 1.5
    spec = np.abs(fft_data)
    spec_sum = np.sum(spec)
    if spec_sum == 0:
        return 3*np.log(1.5) - 1 # the mean of the log freq-axes
    return np.sum(spec*freq_log)/spec_sum


def spectral_variance(data, fft_data):
    """Calculates the variance of the spectrum."""
    return np.var(np.abs(fft_data))


def spectral_skewness(data, fft_data):
    """Calculate the skewness of the spectrum."""
    return scipy.stats.skew(np.abs(fft_data))


def spectral_flatness(data, fft_data):
    """Calculates a measure for the flatness of the spectrum."""
    spec = np.abs(fft_data)
    spec_mean = np.mean(spec)
    spec_gmean = scipy.stats.gmean(spec)
    if spec_mean == 0:
        return 1
    return spec_gmean/spec_mean


def spectral_brightness(data, fft_data):
    """Calculates the ratio between high and low frequencies."""
    spec = np.abs(fft_data)
    weight_vec = np.log(np.linspace(1, 100, len(fft_data)))
    weight_vec = np.pi*weight_vec/weight_vec[-1]
    weight = np.cos(weight_vec)/2 + 0.5
    low_spec_sum = sum(spec*weight)
    high_spec_sum = sum(spec*(1-weight))
    if low_spec_sum == 0:
            return 1 # attention: if signal is a sine at fs/2 this is also hit
    return high_spec_sum/low_spec_sum


def spectral_abs_slope_mean(data, fft_data):
    """Calculates a measure for the variability of the spectrum."""
    spec = np.abs(fft_data)
    slope = np.abs(np.diff(spec))
    return np.mean(slope)
