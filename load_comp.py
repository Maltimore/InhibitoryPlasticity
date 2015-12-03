import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import scipy.ndimage as nd
from numpy.fft import rfft, irfft

def _exp_function(x_vec, mu, scale):
    # catch the case where scale == 0
    if scale == 0:
        y_vec = np.zeros(len(x_vec))
        y_vec[x_vec==mu] = 1
        return y_vec
    elif scale == "ininity":
        # if an infinitely big sensor is desired, return uniform weights
        y_vec = np.ones(len(x_vec))
        y_vec /= np.sum(y_vec)
        return y_vec
    #else, compute normal exponential function
    yvec = np.exp(-np.abs(x_vec - mu) / scale)
    return yvec / np.sum(yvec)


path = os.getcwd()
mresults = pickle.load(open(path + "/results/comparison/malte_results.p", "rb"))
oresults = pickle.load(open(path + "/results/comparison/owen_results_cpp.p", "rb"))
oresults_numpy = pickle.load(open(path + "/results/comparison/owen_results_numpy.p", "rb"))

A = mresults["inh_rates"][:,1]
m_r_hat = mresults["r_hat"][:,1]
o_r_hat = oresults["r_hat"][:,1]
o_numpy_r_hat = oresults_numpy["r_hat"][:,1]

plt.figure(figsize=(10,5))
plt.plot(A)
plt.title("Raw firing rates")

N = 2000
sigma_s = 200
x_NI = 4

x_window = np.arange(0, len(m_r_hat)*x_NI, x_NI)
exp_window = _exp_function(x_window, 4000, sigma_s)
exp_window /= np.sum(exp_window)


plt.figure()
plt.plot(x_window, exp_window)
plt.title("Kernel Malte")


# Owens algorithm for creating the kernel
intercell = x_NI
length = intercell*N
d = np.linspace(intercell-length/2, length/2, N)
d = np.roll(d, int(N/2+1))
k = np.exp(-np.abs(d)/sigma_s)
k /= k.sum()

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


# Owens algorithm for rate sensor
K = rfft(k)
r_hat = irfft(K * rfft(A), N).real
plt.figure(figsize=(10,5))
plt.plot(r_hat)
plt.title("Rate estimate Owen FFT numpy algorithm")
