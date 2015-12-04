import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import scipy.ndimage as nd
from numpy.fft import rfft, irfft
from mytools import _exp_function


path = os.getcwd()
mresults = pickle.load(open(path + "/results/comparison/malte_results.p", "rb"))
oresults = pickle.load(open(path + "/results/comparison/owen_results_cpp.p", "rb"))
oresults_numpy = pickle.load(open(path + "/results/comparison/owen_results_numpy.p", "rb"))

A = mresults["inh_rates"][:,1]
m_r_hat = mresults["r_hat"][:,1]
o_r_hat = oresults["r_hat"][:,1]
o_numpy_r_hat = oresults_numpy["r_hat"][:,1]
m_w = mresults["inhWeights"]
o_w = oresults["inhWeights"]
o_w_n = oresults_numpy["inhWeights"]

plt.figure(figsize=(10,5))
plt.plot(A)
plt.title("Raw firing rates")

NI = oresults["NI"]
sigma_s = oresults["sigma_s"]
x_NI = oresults["x_NI"]

x_window = np.arange(0, len(m_r_hat)*x_NI, x_NI)
exp_window = _exp_function(x_window, int(NI*x_NI/2), sigma_s)
exp_window /= np.sum(exp_window)


plt.figure()
plt.plot(x_window, exp_window)
plt.title("Kernel Malte")


# Owens algorithm for creating the kernel
if sigma_s == "infinity":
    k = np.ones(NI)/NI
elif sigma_s < 1e-3: 
    k = np.zeros(NI)
    k[0] = 1
else:
    intercell = x_NI
    length = intercell*NI
    d = np.linspace(intercell-length/2, length/2, NI)
    d = np.roll(d, int(NI/2+1))
    k = np.exp(-np.abs(d)/sigma_s)
    k /= k.sum()
#k = np.roll(k, int(len(k)/2+1))

plt.figure()
plt.plot(x_window, k)
plt.title("Kernel Owen")

k_rotated = np.roll(k, 1000)

plt.figure()
plt.plot(x_window, k_rotated)
plt.title("Kernel Owen rotated")

plt.figure(figsize=(10,5))
plt.plot(m_r_hat, label="malte")
plt.plot(o_r_hat, label="owen cpp")
plt.plot(o_numpy_r_hat, label="owen numpy")
plt.legend()
plt.title("Rate sensor estimate (real simulation data) Comparison")


convolved = nd.filters.convolve(A, exp_window, mode='wrap')
plt.figure(figsize=(10,5))
plt.plot(convolved)
plt.title("Circular convolution numpy")



plt.figure(figsize=(10,5))
plt.plot(m_r_hat - A)
plt.title("malte - single rates")

plt.figure(figsize=(10,5))
plt.plot(o_r_hat - A)
plt.title("owen cpp - single rates")

plt.figure(figsize=(10,5))
plt.plot(o_numpy_r_hat - A)
plt.title("owen numpy - single rates")

plt.figure(figsize=(10,5))
plt.plot(o_r_hat - o_numpy_r_hat)
plt.title("owen cpp - owen numpy")


plt.figure(figsize=(10,5))
plt.plot(convolved- A)
plt.title("numpy convolve - single rates")