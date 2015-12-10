import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import scipy.ndimage as nd
import scipy.signal as sig
from numpy.fft import rfft, irfft
from mytools import _exp_function


path = os.getcwd()
mresults = pickle.load(open(path + "/results/comparison/malte_results.p", "rb"))
oresults = pickle.load(open(path + "/results/comparison/owen_results_cpp.p", "rb"))
oresults_numpy = pickle.load(open(path + "/results/comparison/owen_results_numpy.p", "rb"))

m_kernel = mresults["kernel_used"]
o_kernel = oresults["kernel_used"]
NI = oresults["NI"]

plt.figure(figsize=(10,5))
plt.plot(m_kernel)
plt.title("Kernel Malte")

plt.figure(figsize=(10,5))
plt.plot(o_kernel)
plt.title("Kernel Owen")

o_k_rotated = np.roll(o_kernel, int(NI/2+1))
plt.figure(figsize=(10,5))
plt.plot(o_k_rotated)
plt.title("Kernel Owen rotated")

print("Difference between the two kernels is: " + 
      str(np.sum(abs(m_kernel - o_k_rotated))))
      
     
toydata = np.zeros(2000)

plt.figure(figsize=(10,5))
plt.plot(toydata)
plt.title("Toydata")

convolved = nd.filters.convolve(toydata, m_kernel, mode='wrap')
plt.figure(figsize=(10,5))
plt.plot(convolved)
plt.title("Circular convolution numpy")


K = rfft(o_kernel)
convolved_fourier = irfft(K * rfft(toydata), NI).real
plt.figure(figsize=(10,5))
plt.plot(convolved_fourier)
plt.title("Circular convolution in fourier space")

print("Difference of convolutions is " + 
      str(np.sum(abs(convolved - convolved_fourier))))