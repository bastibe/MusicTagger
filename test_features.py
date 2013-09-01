import unittest
import numpy as np
import scipy.signal
import features


def zeros(sig_len=1024):
    signal = np.zeros(sig_len)
    return (signal, np.fft.rfft(signal))


def ones(sig_len=1024):
    signal = np.ones(sig_len)
    return (signal, np.fft.rfft(signal))


def sine(periods=1, sig_len=1024):
    # use cosine instead of sine, so sine(sig_len/2, sig_len) returns
    # alternating 1 and -1 instead of zeros
    signal = np.cos(np.linspace(0, periods*2*np.pi, sig_len+1))[:-1]
    return (signal, np.fft.rfft(signal))


def square(periods=1, sig_len=1024):
    signal = scipy.signal.square(np.linspace(0, periods*2*np.pi, sig_len))
    return (signal, np.fft.rfft(signal))


def sawtooth(periods=1, sig_len=1024):
    signal = scipy.signal.sawtooth(np.linspace(0, periods*2*np.pi, sig_len))
    return (signal, np.fft.rfft(signal))


def dirac(sig_len=1024):
    signal = np.concatenate(([1], np.zeros(sig_len-1)))
    return (signal, np.fft.rfft(signal))


class SignalsTestCase(unittest.TestCase):
    def test_zeros_spectrum(self):
        """A zeros spectrum should be zero."""
        signal, fft_signal = zeros(sig_len=1024)
        self.assertTrue(np.all(fft_signal == np.zeros(513)))

    def test_ones_spectrum(self):
        """A ones spectrum should be a dirac."""
        signal, fft_signal = ones(sig_len=1024)
        expected = np.concatenate([[1024], np.zeros(513-1)])
        self.assertTrue(np.all(fft_signal == expected))

    def test_low_sine_spectrum(self):
        """A single sine wave spectrum should be a dirac at index 1."""
        signal, fft_signal = sine(periods=1, sig_len=1024)
        expected = np.concatenate([[0, 512], np.zeros(513-2)])
        self.assertTrue(np.allclose(fft_signal, expected))

    def test_nyquist_sine(self):
        """A nyquist sine should consist of alternating 1 and -1."""
        signal, fft_signal = sine(periods=512, sig_len=1024)
        expected = np.ones(1024)
        expected[1::2] = -1
        self.assertTrue(np.allclose(signal, expected))

    def test_nyquist_sine_spectrum(self):
        """A nyquist sine spectrum should be a dirac at index -1."""
        signal, fft_signal = sine(periods=512, sig_len=1024)
        expected = np.concatenate([np.zeros(513-1), [1024]])
        self.assertTrue(np.allclose(fft_signal, expected))

    def test_dirac_spectrum(self):
        """A dirac spectrum should be ones."""
        signal, fft_signal = dirac(sig_len=1024)
        expected = np.ones(513)
        self.assertTrue(np.allclose(fft_signal, expected))


class RMSTestCase(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(features.rms(*zeros()), 0)

    def test_ones(self):
        self.assertEqual(features.rms(*ones()), 1)

    def test_square(self):
        self.assertEqual(features.rms(*square()), 1)


class PeakTestCase(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(features.peak(*zeros()), 0)

    def test_ones(self):
        self.assertEqual(features.peak(*ones()), 1)

    def test_square(self):
        self.assertEqual(features.peak(*square()), 1)

    def test_saw(self):
        self.assertAlmostEqual(features.peak(*sawtooth(10)), 1)


class CrestTestCase(unittest.TestCase):
    def test_factor_zeros(self):
        self.assertEqual(features.crest_factor(*zeros()), 1)

    def test_factor_ones(self):
        self.assertEqual(features.crest_factor(*ones()), 1)

    def test_factor_rect(self):
        self.assertEqual(features.crest_factor(*square()), 1)

    def test_factor_dirac(self):
        self.assertEqual(features.crest_factor(*dirac()), 32)


class SpectralCentroidTestCase(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(features.spectral_centroid(*zeros()), 0.25)

    def test_ones(self):
        self.assertEqual(features.spectral_centroid(*ones()), 0.0)

    def test_dirac(self):
        # spectrum of a dirac should be constant
        self.assertEqual(features.spectral_centroid(*dirac()), 0.25)

    def test_sine(self):
        # results in peak at freq[1]=1/sig_len with height of 512
        self.assertAlmostEqual(features.spectral_centroid(*sine()), 1/1024)


class LogSpectralCentroidTestCase(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(
            features.log_spectral_centroid(*zeros()), 3*np.log(1.5) - 1)

    def test_dirac(self):
        # result gets closer to analytical value for longer signals.
        self.assertAlmostEqual(
            features.log_spectral_centroid(*dirac(4096)), 3*np.log(1.5) - 1, 4)


class SpectralVarianceTestCase(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(features.spectral_variance(*zeros()), 0)

    def test_dirac(self):
        self.assertEqual(features.spectral_variance(*dirac()), 0)


class SpectralSkewnessTestCase(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(features.spectral_skewness(*zeros()), 0)

    def test_dirac(self):
        self.assertEqual(features.spectral_skewness(*dirac()), 0)

class SpectralFlatnessTestCase(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(features.spectral_flatness(*zeros()), 1)

    def test_dirac(self):
        self.assertEqual(features.spectral_flatness(*dirac()), 1)

    def test_sine(self):
        self.assertAlmostEqual(features.spectral_flatness(*sine()), 0)


class SpectralBrightnessTestCase(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(features.spectral_brightness(*zeros()), 1)

    def test_dc(self):
        self.assertEqual(features.spectral_brightness(*ones()), 0)

    def test_nyquist_sine(self):
        self.assertEqual(features.spectral_brightness(*sine(512)), 1)

    def test_high_noise(self):
        fft_test_data = np.linspace(0, 1, 512)
        test_data = np.fft.irfft(fft_test_data)
        self.assertGreater(
            features.spectral_brightness(test_data, fft_test_data), 1)

    def test_white_noise(self):
        test_data = np.random.randn(1024)
        fft_test_data = np.fft.rfft(test_data)
        self.assertGreater(
            features.spectral_brightness(test_data, fft_test_data), 1)

    def test_low_noise(self):
        test_data = np.random.randn(1024)
        b, a = scipy.signal.butter(8, [0.1, 0.2]) # lowpass filter
        test_data = scipy.signal.lfilter(b, a, test_data)
        fft_test_data = np.fft.rfft(test_data)
        self.assertLess(
            features.spectral_brightness(test_data, fft_test_data), 1)


class SpectralAbsSlopeMeanTestCase(unittest.TestCase):
    def test_zeros(self):
        self.assertEqual(features.spectral_abs_slope_mean(*zeros()), 0)

    def test_dirac(self):
        self.assertEqual(features.spectral_abs_slope_mean(*dirac()), 0)

    def test_sine(self):
        self.assertEqual(features.spectral_abs_slope_mean(*sine()), 2)


if __name__ == '__main__':
    unittest.main()
