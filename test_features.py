import warnings
import numpy as np
import scipy.signal
from unittest import TestCase
import matplotlib.pyplot as plt
import features


def sine(periods=1, sig_len=1024):
    return np.sin(np.linspace(0, periods*2*np.pi, sig_len+1))[:-1]


def square(periods=1, sig_len=1024):
    return scipy.signal.square(np.linspace(0, periods*2*np.pi, sig_len))


def sawtooth(periods=1, sig_len=1024):
    return scipy.signal.sawtooth(np.linspace(0, periods*2*np.pi, sig_len))


def dirac(sig_len=1024):
    return np.concatenate(([1], np.zeros(sig_len-1)))


def test_rms_zeros():
    test_data = np.zeros(1024)
    TestCase().assertEqual(features.rms(test_data, None), 0)
def test_rms_ones():
    test_data = np.ones(1024)
    TestCase().assertEqual(features.rms(test_data, None), 1)
def test_rms_square():
    test_data = square(10)
    TestCase().assertEqual(features.rms(test_data, None), 1)


def test_peak_zeros():
    test_data = np.zeros(1024)
    TestCase().assertEqual(features.peak(test_data, None), 0)
def test_peak_ones():
    test_data = np.ones(1024)
    TestCase().assertEqual(features.peak(test_data, None), 1)
def test_peak_square():
    test_data = square(10)
    TestCase().assertEqual(features.peak(test_data, None), 1)
def test_peak_saw():
    test_data = sawtooth(10)
    TestCase().assertAlmostEqual(features.peak(test_data, None), 1)


def test_crest_factor_zeros():
    test_data = np.zeros(1024)
    TestCase().assertEqual(features.crest_factor(test_data, None), 1)
def test_crest_factor_ones():
    test_data = np.ones(1024)
    TestCase().assertEqual(features.crest_factor(test_data, None), 1)
def test_crest_factor_rect():
    test_data = square(10)
    TestCase().assertEqual(features.crest_factor(test_data, None), 1)
def test_crest_factor_dirac():
    test_data = dirac(1024)
    TestCase().assertEqual(features.crest_factor(test_data, None), 32)


def test_spectral_centroid_zeros():
    test_data = np.zeros(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.spectral_centroid(test_data, fft_test_data), 0.25)
def test_spectral_centroid_dirac():
    test_data = dirac(1024)
    fft_test_data = np.fft.rfft(test_data)
    # spectrum of dirac should be constant
    TestCase().assertEqual(
        features.spectral_centroid(test_data, fft_test_data), 0.25)
def test_spectral_centroid_sine():
    test_data = sine()
    fft_test_data = np.fft.rfft(test_data)
    # results in peak at freq[1]=1/sig_len with height of 512
    TestCase().assertAlmostEqual(
        features.spectral_centroid(test_data, fft_test_data), 1/1024)


def test_log_spectral_centroid_zeros():
    test_data = np.zeros(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.log_spectral_centroid(test_data, fft_test_data), 3*np.log(1.5) - 1)
def test_log_spectral_centroid_dirac():
    test_data = dirac(4096) # result gets closer to analytical value
                            # for longer signals.
    fft_test_data = np.fft.rfft(test_data)
    # due to rounding errors, the result is only approximately correct.
    TestCase().assertAlmostEqual(
        features.log_spectral_centroid(test_data, fft_test_data), 3*np.log(1.5) - 1, 4)


def test_spectral_variance_zeros():
    test_data = np.zeros(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.spectral_variance(test_data, fft_test_data), 0)
def test_spectral_variance_dirac():
    test_data = dirac(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.spectral_variance(test_data, fft_test_data), 0)


def test_spectral_skewness_zeros():
    test_data = np.zeros(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.spectral_skewness(test_data, fft_test_data), 0)
def test_spectral_skewness_dirac():
    test_data = dirac(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.spectral_skewness(test_data, fft_test_data), 0)


def test_spectral_flatness_zeros():
    test_data = np.zeros(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.spectral_flatness(test_data, fft_test_data), 1)
def test_spectral_flatness_dirac():
    test_data = dirac(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.spectral_flatness(test_data, fft_test_data), 1)
def test_spectral_flatness_sine():
    test_data = sine()
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertAlmostEqual(
        features.spectral_flatness(test_data, fft_test_data), 0)


def test_spectral_brightness_zeros():
    test_data = np.zeros(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.spectral_brightness(test_data, fft_test_data), 1)
def test_spectral_brightness_dc():
    test_data = np.ones(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.spectral_brightness(test_data, fft_test_data), 0)
def test_spectral_brightness_nyquist_sine():
    sig_len = 1024
    test_data = np.zeros(sig_len)
    test_data[::2] = 1 # sine at fs/2
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(
        features.spectral_brightness(test_data, fft_test_data), 1)
def test_spectral_brightness_high_noise():
    fft_test_data = np.linspace(0, 1, 512)
    test_data = np.fft.irfft(fft_test_data)
    TestCase().assertGreater(
        features.spectral_brightness(test_data, fft_test_data), 1)
def test_spectral_brightness_white_noise():
    test_data = np.random.randn(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertGreater(
        features.spectral_brightness(test_data, fft_test_data), 1)
def test_spectral_brightness_low_noise():
    test_data = np.random.randn(1024)
    b, a = scipy.signal.butter(8, [0.1, 0.2]) # lowpass filter
    test_data = scipy.signal.lfilter(b, a, test_data)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertLess(
        features.spectral_brightness(test_data, fft_test_data), 1)


def test_spectral_abs_slope_mean_zeros():
    test_data = np.zeros(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(features.spectral_abs_slope_mean(test_data, fft_test_data), 0)
def test_spectral_abs_slope_mean_dirac():
    test_data = dirac(1024)
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(features.spectral_abs_slope_mean(test_data, fft_test_data), 0)
def test_spectral_abs_slope_mean_sine():
    test_data = sine()
    fft_test_data = np.fft.rfft(test_data)
    TestCase().assertEqual(features.spectral_abs_slope_mean(test_data, fft_test_data), 2)
