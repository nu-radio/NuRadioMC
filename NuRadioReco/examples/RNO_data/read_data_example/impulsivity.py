import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import chi2
import ROOT
from scipy.signal import hilbert

class Impulsivity:
    
    def erf_linear(x, A, B):
        
        return (A * erf(x / B) + x) / (A * erf(1. / B) + 1)

    def get_cdf(self, trace):

        # Hilbert transform to get the envelope of the waveform
        hilbert_envelope = np.abs(hilbert(trace))
        # Find the index of the maximum value in the Hilbert envelope
        hill_max_idx = np.argmax(hilbert_envelope)
        hill_max = hilbert_envelope[hill_max_idx]

        # Sorting based on closeness to the maximum index
        closeness = np.abs(np.arange(len(trace)) - hill_max_idx)
        clo_sort_idx = np.argsort(closeness)

        # Sort the Hilbert envelope by closeness to the maximum
        sorted_waveform = hilbert_envelope[clo_sort_idx]

        # Cumulative distribution function (CDF) calculation
        cdf = np.cumsum(sorted_waveform)
        cdf /= np.max(cdf)

        t_frac = np.linspace(0, 1, len(cdf))

        return t_frac, cdf

    def calculate_impulsivity_measures(self, channel_wf):
        result = {}

        t_frac, cdf = self.get_cdf(channel_wf)

        # Calculate impulsivity
        impulsivity = 2 * np.mean(cdf) - 1

        # Linear regression to get slope, intercept, and other statistics
        slope, intercept, r_value, p_value, std_err = linregress(t_frac, cdf)
        cdf_fit = slope * t_frac + intercept

        # Calculate Kolmogorov-Smirnov statistic
        ks = np.max(np.abs(cdf_fit - cdf))

        # Perform erf-linear fit on the CDF
        if(intercept <= 0):
            A0 = 1e-6
        else:
            A0 = intercept/slope
   
        # using ROOT fitter seems to be more robust 
        fErfLinear = ROOT.TF1("fErfLinear", "([0]*erf(x/[1]) + x)/([0]*erf(1/[1])+1)", 0, 1) # define erf_linear in ROOT
        fErfLinear.SetParameters(A0, 1e-2) # set initial guess for parameters [0] and [1]
        fErfLinear.SetParLimits(0, 0, 3*A0) # set bounds on parameter [0]
        fErfLinear.SetParLimits(1, 1e-6, 0.5) # set bound on parameter [1]
        gr = ROOT.TGraph(len(t_frac), t_frac, cdf) # put the data into a TGraph
        gr.Fit(fErfLinear, "SQFM") # perform fit
        A_fit = fErfLinear.GetParameter(0) # get best-fit value on parameter [0] 
        B_fit = fErfLinear.GetParameter(1) # get best-fit value on parameter [1]
        chi2_erf_linear = gr.Chisquare(fErfLinear)   
 
        # Calculate linear chi2 (difference between linear fit and ideal x=y line)
        chi2_linear = np.sum((cdf - t_frac) ** 2)

        # Calculate significance of erf-linear fit over linear fit using Wilks' theorem
        dChi2 = chi2_linear - chi2_erf_linear
        impSig = np.sign(dChi2)*np.sqrt(chi2.ppf(chi2.cdf(abs(dChi2), 2), 1))
        impSig = min(10, impSig) # bound the significance

        # Store the results
        result['impulsivity'] = impulsivity
        result['slope'] = slope
        result['intercept'] = intercept
        result['ks'] = ks  # Kolmogorov-Smirnov statistic
        result['r_value'] = r_value
        result['p_value'] = p_value
        result['std_err'] = std_err
        result['impLinChi2'] = chi2_linear
        result['impErfLinChi2'] = chi2_erf_linear
        result['impSig'] = impSig
        result['impErfA'] = A_fit
        result['impErfB'] = B_fit

        return result
