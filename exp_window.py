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
for neuron_idx in np.arange(N_neurons):
    # extract spike times of just one neuron
    spike_t_current_neuron = spike_times[inhSpikeMon.i == neuron_idx]
    # reverse order of time starting at t
    spike_t_current_neuron = t - spike_t_current_neuron
    # cut out spike times further in the past than the time of the window
    spike_t_current_neuron = spike_t_current_neuron[spike_t_current_neuron < max_t_window]
    # transform spike times to indices in window vector
    spike_indices = spike_t_current_neuron / dt
    spike_indices = spike_indices.astype(int)
    # sum up window values at corresponding spike times
    firing_rate = np.sum(window[spike_indices])
    rate_vector[neuron_idx] = firing_rate
    print("Firing rate estimate is: " + str(firing_rate))
    
plt.figure()
plt.plot(window_time/ms, window)
plt.xlabel("time [ms]")
plt.ylabel("weight")
plt.title("Window integrates to: " + str(np.sum(window*(dt/ms))))
for idx in np.arange(len(spike_indices)):
    plt.axvline(spike_indices[idx]*(dt/ms), ls= '--')
    
matrix_axis = np.floor(np.sqrt(len(rate_vector)))
rate_vector = rate_vector[:matrix_axis**2]
rate_mat = np.reshape(rate_vector, (int(np.sqrt(N_neurons)), -1))

fig, ax = plt.subplots()
heatmap = ax.pcolor(rate_mat, cmap=plt.cm.Greens)