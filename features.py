import warnings
import numpy as np
import scipy.stats
import scipy.signal
from unittest import TestCase
import matplotlib.pyplot as plt


def rms(data, fft_data):
    return np.sqrt(np.mean(data*data))
    
def test_rms_zeros():
    test_data = np.zeros(100)
    assert rms(test_data, None) == 0
def test_rms_ones():
    test_data = np.ones(100)
    assert rms(test_data, None) == 1
def test_rms_rect():
    test_data = scipy.signal.square(np.linspace(0, 10*2*np.pi))
    assert rms(test_data, None) == 1
    

def peak(data, fft_data):
    return np.max(np.abs(data))    

def test_peak_zeros():
    test_data = np.zeros(10)
    assert peak(test_data, None) == 0
def test_peak_ones():
    test_data = np.ones(10)
    assert peak(test_data, None) == 1
def test_peak_rect():
    test_data = scipy.signal.square(np.linspace(0, 10*2*np.pi))
    assert peak(test_data, None) == 1
def test_peak_saw():
    test_data = scipy.signal.sawtooth(np.linspace(0, 10*2*np.pi))
    test = TestCase()
    test.assertAlmostEqual(peak(test_data, None), 1)


def crest(data, fft_data):
    peak_val = peak(data, fft_data)
    rms_val = rms(data, fft_data)
    if peak_val == 0:
        return 1
    return peak_val/rms_val
    
def test_crest_zeros():
    test_data = np.zeros(100)
    assert crest(test_data, None) == 1
def test_crest_ones():
    test_data = np.ones(100)
    assert crest(test_data, None) == 1
def test_crest_rect():
    test_data = scipy.signal.square(np.linspace(0, 10*2*np.pi))
    assert crest(test_data, None) == 1
def test_crest_dirac():
    test_data = np.zeros(100)
    test_data[0] = 1
    assert crest(test_data, None) == 10
    
        
def spectral_centroid(data, fft_data):
    freq = np.fft.fftfreq(len(data))[:len(fft_data)]
    freq[-1] += 1 # last element needs to be 0.5 not -0.5
    spec = np.abs(fft_data)
    spec_sum = np.sum(spec)
    if spec_sum == 0:
        return 0.25
    centroid = np.sum(spec*freq)/spec_sum
    return centroid

def test_spectral_centroid_zeros():
    test_data = np.zeros(100)
    fft_test_data = np.fft.rfft(test_data)
    assert spectral_centroid(test_data, fft_test_data) == 0.25
def test_spectral_centroid_dirac():
    test_data = np.zeros(100)
    test_data[0] = 1
    fft_test_data = np.fft.rfft(test_data)
    assert spectral_centroid(test_data, fft_test_data) == 0.25 # spectrum of dirac should be constant
def test_spectral_centroid_sine():
    sig_len = 1024
    test_data = np.sin(np.linspace(0, 2*np.pi, sig_len+1))[:-1]
    fft_test_data = np.fft.rfft(test_data)
    # results in peak at freq[1]=1/sig_len with height of 500
    test = TestCase()
    test.assertAlmostEqual(spectral_centroid(test_data, fft_test_data), 1/sig_len)


def log_spectral_centroid(data, fft_data):
    freq = np.fft.fftfreq(len(data))[:len(fft_data)]
    freq[-1] += 1 # last element needs to be 0.5 not -0.5
    freq_log = np.log(freq+1) # linear range into 1 to 1.5 
    spec = np.abs(fft_data)
    spec_sum = np.sum(spec)
    if spec_sum == 0:
        return 3*np.log(1.5) - 1 # the mean of the log freq-axes
    return np.sum(spec*freq_log)/spec_sum

def test_log_spectral_centroid_zeros():
    test_data = np.zeros(128)
    fft_test_data = np.fft.rfft(test_data)
    assert log_spectral_centroid(test_data, fft_test_data) == 3*np.log(1.5) - 1
def test_log_spectral_centroid_dirac():
    test_data = np.zeros(4096)
    test_data[0] = 1
    fft_test_data = np.fft.rfft(test_data)
    test = TestCase()
    test.assertAlmostEqual(log_spectral_centroid(test_data, fft_test_data), 3*np.log(1.5) - 1, 4)
    
    
def spectral_variance(data, fft_data):
    return np.var(np.abs(fft_data))
    
def test_spectral_variance_zeros():
    test_data = np.zeros(128)
    fft_test_data = np.fft.rfft(test_data)
    assert spectral_variance(test_data, fft_test_data) == 0
def test_spectral_variance_dirac():
    test_data = np.zeros(128)
    test_data[0] = 1
    fft_test_data = np.fft.rfft(test_data)
    assert spectral_variance(test_data, fft_test_data) == 0
    
    
def spectral_skewness(data, fft_data):
    return scipy.stats.skew(np.abs(fft_data))
    
def test_spectral_skewness_zeros():
    test_data = np.zeros(128)
    fft_test_data = np.fft.rfft(test_data)
    # TODO look into scipy.stats.skew (stats.py: 1067)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        assert spectral_skewness(test_data, fft_test_data) == 0
def test_spectral_skewness_dirac():
    test_data = np.zeros(128)
    test_data[0] = 1
    fft_test_data = np.fft.rfft(test_data)
    assert spectral_skewness(test_data, fft_test_data) == 0    
    
    
def spectral_flatness(data, fft_data):
    spec = np.abs(fft_data)
    spec_mean = np.mean(spec)
    spec_gmean = scipy.stats.gmean(spec)
    if spec_mean == 0:
        return 1
    return spec_gmean/spec_mean
    
def test_spectral_flatness_zeros():
    test_data = np.zeros(128)
    fft_test_data = np.fft.rfft(test_data)
    assert spectral_flatness(test_data, fft_test_data) == 1
def test_spectral_flatness_dirac():
    test_data = np.zeros(128)
    test_data[0] = 1
    fft_test_data = np.fft.rfft(test_data)
    assert spectral_flatness(test_data, fft_test_data) == 1
def test_spectral_flatness_sine():
    sig_len = 1024
    test_data = np.sin(np.linspace(0, 2*np.pi, sig_len+1))[:-1]
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertAlmostEqual(spectral_flatness(test_data, fft_test_data), 0)
    

def spectral_brightness(data, fft_data):
    spec = np.abs(fft_data)
    weight_vec = np.log(np.linspace(1, 100, len(fft_data)))
    weight_vec = np.pi*weight_vec/weight_vec[-1]
    weight = np.cos(weight_vec)/2 + 0.5
    low_spec_sum = sum(spec*weight)
    high_spec_sum = sum(spec*(1-weight))
    if low_spec_sum == 0:
            return 1 # attention: if signal is a sine at fs/2 this is also hit
    return high_spec_sum/low_spec_sum

def test_spectral_brightness_zeros():
    test_data = np.zeros(128)
    fft_test_data = np.fft.rfft(test_data)
    assert spectral_brightness(test_data, fft_test_data) == 1
def test_spectral_brightness_dc():
    test_data = np.ones(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(spectral_brightness(test_data, fft_test_data), 0)
def test_spectral_brightness_nyquist_sine():
    sig_len = 1024
    test_data = np.zeros(sig_len)
    test_data[::2] = 1 # sine at fs/2
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(spectral_brightness(test_data, fft_test_data), 1)
def test_spectral_brightness_high_noise():
    fft_test_data = np.linspace(0, 1, 512)
    test_data = np.fft.irfft(fft_test_data)
    TestCase().assertGreater(spectral_brightness(test_data, fft_test_data), 1)
def test_spectral_brightness_white_noise():
    test_data = np.random.randn(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertGreater(spectral_brightness(test_data, fft_test_data), 1)
def test_spectral_brightness_low_noise():
    test_data = np.random.randn(1024)
    b, a = scipy.signal.butter(8, [0.1, 0.2])
    test_data = scipy.signal.lfilter(b, a, test_data)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertLess(spectral_brightness(test_data, fft_test_data), 1)
    
    
def spectral_abs_slope_mean(data, fft_data):
    spec = np.abs(fft_data)
    slope = np.abs(np.diff(spec))
    return np.mean(slope)
    
    