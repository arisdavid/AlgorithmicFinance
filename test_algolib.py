"""
Developer: Aris David
Date: 25.06.2018
Description: For Unit testing algolib

"""

import unittest
from lib.algolib import AlgoLib
import numpy as np


class TestAlgoLib(unittest.TestCase):
   
    #Test Monte Carlo Geometric Brownian Motion
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
        
    #Test KMV Model
    def test_kmv_edf(self):
        enterprise_value = 1000
        short_term_debt = 400
        long_term_debt = 400
        mu = 0.2
        sigma = 0.25 
        period = 1
        edf = AlgoLib.kmv(enterprise_value, short_term_debt, long_term_debt, mu, sigma, period)
        self.assertEqual(edf, 0.0032808908834941666)

if __name__ == '__main__':
    unittest.main()
