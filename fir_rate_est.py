from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters


def exponential_window(tau, dt):
    max_t = 5 * tau
    time = np.arange(0, max_t/ms, dt/ms) * ms
    window = 1 / tau * np.exp(-time/(tau))
    return time, window



N_neurons = len(inhSpikeMon.spike_trains())
tau = 100*ms
dt = .1*ms
window_time, window = exponential_window(tau, dt)
spike_times = inhSpikeMon.t
t = 3000 * ms #################### !! Later this has to be an argument to the function
max_t_window = window_time[-1]
rate_vector = np.zeros(N_neurons)

### ESTIMATING FIRING RATE BY weighting spike times with window ###############
for neuron_idx in np.arange(N_neurons):
    # extract spike times of just one neuron
    spike_train = spike_times[inhSpikeMon.i == neuron_idx]
    # reverse order of time starting at t
    spike_train = t - spike_train
    # cut out spike times further in the past than the time of the window
    spike_train = spike_train[spike_train < max_t_window]
    # transform spike times to indices in window vector
    spike_indices = spike_train / dt
    spike_indices = spike_indices.astype(int)
    # sum up window values at corresponding spike times
    rate_vector[neuron_idx] = np.sum(window[spike_indices])

# exponential  window plot
plt.figure()
plt.plot(window_time/ms, window)
plt.xlabel("time [ms]")
plt.ylabel("weight")
plt.title("Window integrates to: " + str(np.sum(window*(dt/ms))))
for idx in np.arange(len(spike_indices)):
    plt.axvline(spike_indices[idx]*(dt/ms), ls= '--')

# firing rate plot as matrix
matrix_axis = np.floor(np.sqrt(len(rate_vector)))
rate_vector = rate_vector[:matrix_axis**2]
rate_mat = np.reshape(rate_vector, (int(np.sqrt(N_neurons)), -1))
fig, ax = plt.subplots()
heatmap = ax.pcolor(rate_mat, cmap=plt.cm.Greens)
plt.title("Firing rate estimated with exponential window")
plt.xticks([]); plt.yticks([])





### ESTIMATING FIRING RATE BY JUST COUNTING SPIKES IN THE LAST X ms ###########
max_time = 500 * ms
rate_vector2 = np.zeros(N_neurons)
for neuron_idx in np.arange(N_neurons):
    # extract spike times of just one neuron
    spike_train = spike_times[inhSpikeMon.i == neuron_idx]
    # reverse order of time starting at t
    spike_train = t - spike_train
    # cut out spike times further in the past than the time of the window
    spike_train = spike_train[spike_train < max_time]
    # estimate firing rate by dividing number of spikes by window time
    rate_vector2[neuron_idx] = len(spike_train) / (max_time/second)



# firing rate plot as matrix
matrix_axis = np.floor(np.sqrt(len(rate_vector2)))
rate_vector2 = rate_vector2[:matrix_axis**2]
rate_mat2 = np.reshape(rate_vector2, (int(np.sqrt(N_neurons)), -1))
fig, ax = plt.subplots()
heatmap = ax.pcolor(rate_mat2, cmap=plt.cm.Greens)
plt.title("Firing rate estimated with counting spikes")
plt.xticks([]); plt.yticks([])