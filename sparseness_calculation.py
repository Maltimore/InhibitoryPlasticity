import numpy as np



def compute_sparseness(rate_holder):
    rates = rate_holder[:,-1]    
    NI = len(rates)
    sparseness = ((np.sum(rates)/NI)**2) / (np.sum(rates**2)/NI)
    return sparseness

sparseness = compute_sparseness(rate_holder)