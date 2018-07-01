"""
Developer: Aris David
Date: 25.06.2017
Description: Quantitative and numerical functions

"""

import numpy as np
import math
import numpy.matlib as matlib
from scipy.stats import norm

class AlgoLib():
    
    
    @staticmethod
    def gbm_monte_carlo(s0, sigma, mu, nPer, nTDays, nSim):
        """
        Geometric Brownian Motion Monte Carlo Simulation
        #s0 - starting price
        #sigma - annual volatility
        #mu - annual expected return
        #nPer - number of forecast period in days
        #nTDays - number of trading days in one year  
        #nSim - number of simulations
        returns simulated asset path
        """
        
        ''' Step 1 - Calculate the Deterministic component - drift    
        Alternative drift 1 - supporting random walk theory
        drift = 0     
        Alternative drift 2 - 
        drift = risk_free_rate - (0.5 * sigma**2)
        '''
        
        #Industry standard for drift
        mu = mu/nTDays  #Daily return 
        sigma = sigma/math.sqrt(nTDays) #Daily volatility
        drift = mu - (0.5 * sigma**2)
        
        
        ''' Step 2 - Create a matrix of stochastic component - random shock ''' 
        z = np.random.normal(0, 1, (nSim, nPer))
        log_ret = drift + (sigma * z)
        
        ''' Compound return using vectorize method 
        LN(Today/Yesterday) = drift + random shock * sigma 
        '''       
        compounded_ret = np.cumsum(log_ret, axis=1)   
        asset_path = s0 + (s0 * compounded_ret)
        
        #Include starting value
        s0 = matlib.repmat(s0, nSim, 1)
        asset_path = np.concatenate((s0, asset_path), axis=1)
        asset_path *= (asset_path > 0) #set negative to zero
                                  
        return (asset_path)

    @staticmethod
    def kmv(ev, stDebt, ltDebt, mu, sigma, period = 1):
        
        """
            KMV model is based on the structural approach to calculate EDF (Expected Default Frequency)
            #ev = enterprise value
            #stDebt = short term debt
            #ltDebt = long term debt
            #mu = expected growth after 1 year
            #sigma = annualized volatility 
            #period = period in years
        """
         #Calcualte default point
        default_point = stDebt + (0.5 * ltDebt)
         
         
        numer = math.log(ev/default_point) + (mu - math.pow(sigma, 2)/2) * period
        denom = sigma * period
        stdDev = numer/denom                 
        edf = norm.cdf(-stdDev)
        
        return edf


