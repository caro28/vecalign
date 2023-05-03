import unittest
from score_newstrict_v5 import _precision


class TestNewStrictScoring(unittest.TestCase):
    def test_precision1a(self):
        goldalign = [([0,1,2,3], [2,3])]
        testalign = [([0], [2]), ([1], [3])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        expected = 0
        print(f"final supersets are: {supersets}")
        self.assertEqual(results, expected)

    def test_precision1b(self):
        goldalign = [([0,1,2,3], [2,3])]
        testalign = [([0], [2]), ([1,2,3], [3])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        print(f"final supersets are: {supersets}")
        expected = 1
        self.assertEqual(results, expected)

    def test_precision1c(self):
        goldalign = [([0,1,2,3], [2,3])]
        testalign = [([0], [2]), ([1,2,3], [3]), ([3], [4])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        print(f"final supersets were {supersets}")
        expected = 0
        self.assertEqual(results, expected)
    
    def test_precision2a(self):
        goldalign = [([0,1,2], [2]), ([3], [3])]
        testalign = [([0,1,2], [2]), ([3], [3])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        print(f"final supersets were: {supersets}")
        expected = 2
        self.assertEqual(results, expected)
    
    def test_precision2b(self):
        goldalign = [([0,1,2], [2]), ([3], [3])]
        testalign = [([0,1,2,3], [2,3])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        print(f"final supersets were: {supersets}")
        expected = 0
        self.assertEqual(results, expected)  

    def test_precision2c(self):
        goldalign = [([0,1,2], [2]), ([3], [3])]
        testalign = [([0], [2]), ([1,2,3], [3])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        print(f"final supersets were: {supersets}")
        expected = 0
        self.assertEqual(results, expected)
    
    def test_precision3a(self):
        goldalign = [([0,1,3], [2,3]), ([2], [])]
        testalign = [([0], [3]), ([1,3], [2])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        print(f"final supersets were: {supersets}")
        expected = 1
        self.assertEqual(results, expected)
    
    def test_precision3b(self):
        goldalign = [([0,1,3], [2,3]), ([2], [])]
        testalign = [([2], [])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        print(f"final supersets were: {supersets}")
        expected = 1
        self.assertEqual(results, expected)
    
    def test_precision3c(self):
        goldalign = [([0,1,3], [2,3]), ([2], [])]
        testalign = [([0], [3]), ([1,3], [2]), ([2], [])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        print(f"final supersets were: {supersets}")
        expected = 2
        self.assertEqual(results, expected)
    
    def test_precision3d(self):
        goldalign = [([0,1,3], [2,3]), ([2], [])]
        testalign = [([0,1,2,3], [2,3])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        print(f"final supersets were: {supersets}")
        expected = 0
        self.assertEqual(results, expected)

    def test_precision4a(self):
        goldalign = [([0,1,2,3], [2,3,4,5])]
        testalign = [([0], [2]), ([1], [3]), ([2], [4]), ([3], [5])]
        results, supersets = _precision(goldalign=goldalign, testalign=testalign)
        print(f"final supersets were: {supersets}")
        expected = 1
        self.assertEqual(results, expected)

if __name__ == "__main__":
    unittest.main(verbosity=5)