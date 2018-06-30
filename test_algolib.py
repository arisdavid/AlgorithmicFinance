"""
Developer: Aris David
Date: 25.06.2018
Description: For Unit testing algolib

"""

import unittest
from lib.algolib import AlgoLib
import numpy as np


class TestAlgoLib(unittest.TestCase):
   
    
    def test_monte_carlo_gbm(self):
        s0 = 100
        sigma = 0.18
        mu = 0.08
        nPer = 250
        nTDays = 250
        nSim = 1000        
        assetPath = AlgoLib.gbm_monte_carlo(s0, sigma, mu, nPer, nTDays, nSim)
        self.assertEqual(np.size(assetPath, axis=0), nSim)
        self.assertEqual(np.size(assetPath, axis=1), nPer+1)

if __name__ == '__main__':
    unittest.main()
