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

A_m = mresults["inh_rates"][:,1]
A_o = oresults["inh_rates"][:,1]
A_on = oresults_numpy["inh_rates"][:,1]
m_r_hat = mresults["r_hat"][:,1]
o_r_hat = oresults["r_hat"][:,1]
o_numpy_r_hat = oresults_numpy["r_hat"][:,1]
m_w = mresults["inhWeights"]
o_w = oresults["inhWeights"]
o_w_n = oresults_numpy["inhWeights"]
m_kernel = mresults["kernel_used"]
o_kernel = oresults["kernel_used"]

plt.figure(figsize=(10,5))
plt.plot(A_m)
plt.title("Raw firing rates")

NI = oresults["NI"]
sigma_s = oresults["sigma_s"]
x_NI = oresults["x_NI"]

x_window = np.arange(0, len(m_r_hat)*x_NI, x_NI)


plt.figure(figsize=(10,5))
plt.plot(x_window, m_kernel)
plt.title("Kernel Malte")

plt.figure(figsize=(10,5))
plt.plot(x_window, o_kernel)
plt.title("Kernel Owen")

o_k_rotated = np.roll(o_kernel, int(NI/2+1))

plt.figure(figsize=(10,5))
plt.plot(x_window, o_k_rotated)
plt.title("Kernel Owen rotated")

print("Difference between the two kernels is: " + 
      str(np.sum(abs(m_kernel - o_k_rotated))))

plt.figure(figsize=(10,5))
plt.plot(m_r_hat, label="malte")
plt.plot(o_r_hat, label="owen cpp")
plt.plot(o_numpy_r_hat, label="owen numpy")
plt.legend()
plt.title("Rate sensor estimate (real simulation data) Comparison")


convolved = nd.filters.convolve(A_m, m_kernel, mode='wrap')
plt.figure(figsize=(10,5))
plt.plot(convolved)
plt.title("Circular convolution numpy")

plt.figure(figsize=(10,5))
plt.plot(o_r_hat - o_numpy_r_hat)
plt.title("owen cpp - owen numpy")




plt.figure()
plt.plot(A_m - A_o)


#plt.figure(figsize=(10,5))
#plt.plot(m_r_hat - A_m)
#plt.title("malte - single rates")
#
#plt.figure(figsize=(10,5))
#plt.plot(o_r_hat - A_o)
#plt.title("owen cpp - single rates")
#
#plt.figure(figsize=(10,5))
#plt.plot(o_numpy_r_hat - A_on)
#plt.title("owen numpy - single rates")
#
#plt.figure(figsize=(10,5))
#plt.plot(convolved- A_m)
#plt.title("numpy convolve - single rates")


#goodness = np.empty(NI)
#A_complex = rfft(A_on)
#for idx in np.arange(NI):
#    k_temp = np.roll(exp_window, idx)
#    K = rfft(k_temp)
#    temp_r_hat = irfft(K * A_complex, NI).real
#    temp_goodness =np.sum(abs(temp_r_hat - m_r_hat))
#    goodness[idx] = temp_goodness
#    if temp_goodness < 100:
#        print(str(idx))
#        print("Found a good kernel")
#        break
#    
#plt.figure()
#plt.plot(goodness)
#plt.title("Amount of error between result of FFT algorithm and normal conv")
#plt.xlabel("Amount by wich kernel was rolled")