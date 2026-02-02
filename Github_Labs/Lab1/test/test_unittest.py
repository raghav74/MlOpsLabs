import sys
import os
import unittest

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import data_processor


class TestDataProcessor(unittest.TestCase):

    def test_normalize(self):
        self.assertEqual(data_processor.normalize([0, 50, 100]), [0.0, 0.5, 1.0])
        self.assertEqual(data_processor.normalize([10, 10, 10]), [0.0, 0.0, 0.0])
        self.assertEqual(data_processor.normalize([1, 2, 3, 4, 5]), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(data_processor.normalize([-10, 0, 10]), [0.0, 0.5, 1.0])

    def test_standardize(self):
        result = data_processor.standardize([1, 2, 3, 4, 5])
        self.assertAlmostEqual(sum(result), 0.0, places=10)  # mean should be ~0
        self.assertEqual(data_processor.standardize([5, 5, 5]), [0.0, 0.0, 0.0])

    def test_fill_missing(self):
        self.assertEqual(data_processor.fill_missing([1, None, 3]), [1, 0, 3])
        self.assertEqual(data_processor.fill_missing([None, None], fill_value=-1), [-1, -1])
        self.assertEqual(data_processor.fill_missing([1, 2, 3]), [1, 2, 3])
        self.assertEqual(data_processor.fill_missing([]), [])

    def test_compute_statistics(self):
        stats = data_processor.compute_statistics([1, 2, 3, 4, 5])
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["min"], 1)
        self.assertEqual(stats["max"], 5)
        self.assertEqual(stats["count"], 5)
        
        stats2 = data_processor.compute_statistics([10, 20, 30])
        self.assertEqual(stats2["mean"], 20.0)
        self.assertEqual(stats2["min"], 10)
        self.assertEqual(stats2["max"], 30)


if __name__ == '__main__':
    unittest.main()
