import numpy as np
import scipy.stats


def rms(data, fft_data):
    return np.sqrt(np.mean(data*data))


def peak(data, fft_data):
    return np.max(np.abs(data))

    
def crest(data, fft_data):
    return peak(data, fft_data)/rms(data, fft_data)

    
def spectral_centroid(data, fft_data):
    freq = np.fft.fftfreq(len(data))[:len(data)/2+1]
    spec = np.abs(fft_data)
    spec_sum = np.sum(spec)
    if spec_sum == 0:
        return 0.25
    centroid = np.sum(spec*freq)/spec_sum
    return centroid


def log_spectral_centroid(data, fft_data):
    freq = np.fft.fftfreq(len(data))[:len(data)/2+1]
    freq = freq+1
    spec = np.abs(fft_data)
    log_centroid = np.sum(spec*np.log(freq))/np.sum(spec)
    return log_centroid

    
def spectral_variance(data, fft_data):
    return np.var(np.abs(fft_data))
    
    
def spectral_skewness(data, fft_data):
    return scipy.stats.skew(np.abs(fft_data))
    
    
def spectral_flatness(data, fft_data):
    spec = np.abs(fft_data)
    return scipy.stats.gmean(spec)/np.mean(spec)
    

def spectral_brightness(data, fft_data):
    centroid = spectral_centroid(data, fft_data)
    centroid_idx = int(centroid*2*len(fft_data))
    spec = np.abs(fft_data)
    return sum(spec[:centroid_idx])/sum(spec[centroid_idx:])


def spectral_abs_slope_mean(data, fft_data):
    spec = np.abs(fft_data)
    slope = np.abs(np.diff(spec))
    return np.mean(slope)