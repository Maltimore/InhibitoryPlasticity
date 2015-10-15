from brian2 import *
import numpy as np
import scipy.ndimage.filters as filters
from scipy.signal import exponential, gaussian

###############################################################################
###############################################################################
# In order for this script to work, a PopulationRateMonitor called rateMon
# must exist (and be filled with values obviously.)
###############################################################################
###############################################################################

tau = 100*ms
dt = .1*ms
quickrate = rateMon.rate/Hz
# Plot rate as estimated from RateMonitor
plt.plot(rateMon.t/ms, quickrate)
plt.xlabel("time [ms]")
plt.ylabel("rate [Hz]")
plt.title("Plot rate as estimated from RateMonitor")

# Filter with gaussian function
result = filters.gaussian_filter1d(quickrate, tau/dt, mode='reflect')
plt.figure()
plt.plot(rateMon.t/ms, result)
plt.xlabel("time [ms]")
plt.ylabel("rate [Hz]")
plt.ylim([10, 25])
plt.title("Filtered with gaussian (SD = " + str(tau) + ")")

print(str(result[-1]))

# Determine number of elements in window and time vector
M = 5*tau/dt # window size
all_time = np.arange(-M/2*(dt/ms), M/2*(dt/ms), dt/ms)
current_window = np.ones(M)
mask = all_time > 0
time = all_time[mask]

# Create a half rectangular window
current_window = np.ones(M)
time = all_time
current_window[~mask] = 0
current_window = current_window / np.sum(current_window)
result = np.convolve(current_window, quickrate, mode='same')
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].plot(time, current_window)
axes[0].set_xlabel("time [ms]")
axes[0].set_ylabel("filtering weight")
axes[0].set_title("Rectangular window")
axes[1].plot(rateMon.t/ms, result)
axes[1].set_xlabel("time [ms]")
axes[1].set_ylabel("rate [Hz]")
axes[1].set_title("Filtered firing rate")
axes[2].plot(rateMon.t/ms, result)
axes[2].set_xlabel("time [ms]")
axes[2].set_ylabel("rate [Hz]")
axes[2].set_xlim([250, 1000])
axes[2].set_ylim([10, 25])
axes[2].set_title("Filtered firing rate")


# Create negative exponential filter
current_window = exponential(M, tau=tau/dt)
time = all_time
current_window[~mask] = 0
current_window = current_window / np.sum(current_window)
result = np.convolve(current_window, quickrate, mode='same')
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].plot(time, current_window)
axes[0].set_xlabel("time [ms]")
axes[0].set_ylabel("filtering weight")
axes[0].set_title("Negative exponential window")
axes[1].plot(rateMon.t/ms, result)
axes[1].set_xlabel("time [ms]")
axes[1].set_ylabel("rate [Hz]")
axes[1].set_title("Filtered firing rate")
axes[2].plot(rateMon.t/ms, result)
axes[2].set_xlabel("time [ms]")
axes[2].set_ylabel("rate [Hz]")
axes[2].set_xlim([250, 1000])
axes[2].set_ylim([10, 25])
axes[2].set_title("Filtered firing rate")

# Create gaussian filter
current_window = gaussian(M, std=tau/dt)
current_window[~mask] = 0
current_window = current_window / np.sum(current_window)
time = all_time
result = np.convolve(current_window, quickrate, mode='same')
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].plot(time, current_window)
axes[0].set_xlabel("time [ms]")
axes[0].set_ylabel("filtering weight")
axes[0].set_title("Half gaussian window")
axes[1].plot(rateMon.t/ms, result)
axes[1].set_xlabel("time [ms]")
axes[1].set_ylabel("rate [Hz]")
axes[1].set_title("Filtered firing rate")
axes[2].plot(rateMon.t/ms, result)
axes[2].set_xlabel("time [ms]")
axes[2].set_ylabel("rate [Hz]")
axes[2].set_xlim([250, 1000])
axes[2].set_ylim([10, 25])
axes[2].set_title("Filtered firing rate")

sum_up = 0
for idx in np.arange(0, int(M/2), 1):
    sum_up += current_window[idx + int(M/2)] * quickrate[-idx-1]
print(str(sum_up))
print(result[-1])

# Create gaussian filter
current_window = gaussian(M, std=tau/dt)
#current_window[~mask] = 0
current_window = current_window / np.sum(current_window)
time = all_time
result = np.convolve(current_window, quickrate, mode='same')
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].plot(time, current_window)
axes[0].set_xlabel("time [ms]")
axes[0].set_ylabel("filtering weight")
axes[0].set_title("Whole gaussian window")
axes[1].plot(rateMon.t/ms, result)
axes[1].set_xlabel("time [ms]")
axes[1].set_ylabel("rate [Hz]")
axes[1].set_title("Filtered firing rate")
axes[2].plot(rateMon.t/ms, result)
axes[2].set_xlabel("time [ms]")
axes[2].set_ylabel("rate [Hz]")
axes[2].set_xlim([250, 1000])
axes[2].set_ylim([10, 25])
axes[2].set_title("Filtered firing rate")


