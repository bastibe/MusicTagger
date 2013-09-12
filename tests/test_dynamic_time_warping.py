import unittest
import numpy as np
import sys
sys.path.append("..")
import dynamic_time_warping as dtw


class DTWTestCase(unittest.TestCase):
    def setUp(self):
        self.signal1 = np.random.randn(50, 10)
        self.signal2 = np.random.randn(40, 10)

    def test_implementations(self):
        """The C and Python implementations should be equal."""
        self.assertAlmostEqual(dtw.dtw_distance(self.signal1, self.signal2),
                               dtw.dtw_distance_c(self.signal1, self.signal2),
                               places=5)

    def test_dtw_same_signal(self):
        """The DTW distance of equal signals should be zero."""
        self.assertEqual(dtw.dtw_distance_c(self.signal1, self.signal1), 0)

    def test_dtw_stretched_signal(self):
        """The DTW distance of two stretched signals should be zero."""
        stretched = self.signal1
        for idx in np.random.randint(50, size=10):
            stretched = np.insert(stretched, idx, stretched[idx], axis=0)
        self.assertEqual(self.signal1.shape[1], stretched.shape[1])
        self.assertEqual(self.signal1.shape[0], stretched.shape[0]-10)
        self.assertEqual(dtw.dtw_distance_c(self.signal1, stretched), 0)

    def test_dtw_argument_order(self):
        """The DTW distance of two signals shoulr be independant of argument order."""
        self.assertEqual(dtw.dtw_distance_c(self.signal1, self.signal2),
                         dtw.dtw_distance_c(self.signal2, self.signal1))


if __name__ == '__main__':
    unittest.main()
