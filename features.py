import numpy as np

def rms(data, fft_data):
    return np.sqrt(np.mean(data*data))


def peak(data, fft_data):
    return np.max(np.abs(data))


def spectral_centroid(data, fft_data):
    freq = np.fft.fftfreq(len(data))[:len(data)/2+1]
    spec = np.abs(fft_data)
    centroid = np.sum(spec*freq)/np.sum(spec)
    return centroid


def log_spectral_centroid(data, fft_data):
    freq = np.fft.fftfreq(len(data))[:len(data)/2+1]
    freq = freq+1
    spec = np.abs(fft_data)
    log_centroid = np.sum(spec*np.log(freq))/np.sum(spec)
    return log_centroid
