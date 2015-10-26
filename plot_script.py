from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import mytools
import imp
imp.reload(mytools)

plot_n_weights = 200
plot_n_rates = 200

def create_plots(SpikeMon, inhSpikeMon, rate_interval, w_holder, rho_0,
                 simtime, dt):
    print("Creating plots..")
    N_inh_neurons = len(inhSpikeMon.spike_trains())    
#    # all spikes
#    plt.figure()
#    plt.plot(SpikeMon.t/ms, SpikeMon.i, '.k', markersize=.1)
#    plt.xlabel("Time (ms)")
#    plt.ylabel("Neuron index")
#
#    # first show_n inhibitory neurons spikes
#    show_n = 20
#    plt.figure()
#    plt.plot(inhSpikeMon.t/ms, inhSpikeMon.i, '.k')
#    plt.ylim([0, show_n])
#    plt.xlabel("Time (ms)")
#    plt.ylabel("Neuron index")
#    plt.title("Spikes in the first " + str(show_n) + " inhibitory neurons")
    
    # inhibitory firing rate and weights over time
    # draw randomly plot_n_weights from the weights matrix
    w_idxes = np.random.uniform(w_holder.shape[0], size=plot_n_weights)
    w_idxes = w_idxes.astype(int)
    w_streams = w_holder[w_idxes, :]
    avg_w_stream = np.average(w_streams, axis=0)
    
    print("Estimating single firing rates..", end="", flush=True)
    _, rate_vector = mytools.estimate_single_firing_rates(inhSpikeMon, 
                                                    rate_interval,
                                                    simtime,
                                                    t_min= 0*ms,
                                                    N_neurons = plot_n_rates)
    print(" Done.", flush=True)
    times, rates = mytools.estimate_pop_firing_rate(inhSpikeMon, rate_interval,
                                                    simtime)
    w_times = np.hstack((0*ms, times))
    print(times)    
    print(w_times)
    print(avg_w_stream)
    print(len(w_times))
    print(len(avg_w_stream))
    fig, axes = plt.subplots(2, figsize=(15, 10))
    axes[0].plot(times/second, rate_vector.T, color="red",
                 alpha=.2, linewidth=.3)
    axes[0].plot(times/second, rates, color="red", label="firing_rate")
    axes[0].hlines(rho_0, 0, times[-1], linestyles="--")
    axes[0].set_xlim([0, times[-1]])
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("firing rate [Hz]")
    axes[0].set_title(str(plot_n_rates) + \
                      " randomly selected firing rates estimated every " + \
                      str(rate_interval))
    axes[1].plot(w_times/second, w_streams.T, color="gray", alpha=.2,
                 linewidth=.3)
    axes[1].plot(w_times/second, avg_w_stream, color="black")
    axes[1].hlines(0, 0, times[-1], linestyles="--")
    axes[1].set_ylim([-1, np.amax(w_streams)+10])
    axes[1].set_xlim([0, times[-1]])
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("Inh to exc weight")
    axes[1].set_title(str(plot_n_weights) + \
                      " randomly selected inh-to-exc weights")

    
    # firing rate plot as matrix
    _, rate_vector = mytools.estimate_single_firing_rates(inhSpikeMon, 
                                                          rate_interval,
                                                          simtime)
    matrix_axis = np.floor(np.sqrt(len(rate_vector)))
    rate_vector = rate_vector[:matrix_axis**2]
    rate_mat = np.reshape(rate_vector, (int(np.sqrt(N_inh_neurons)), -1))
    fig, ax = plt.subplots()
    ax.pcolor(rate_mat, cmap="Reds")
    plt.title("Inh firing rate estimated with counting spikes")
    plt.xticks([]); plt.yticks([]);
    
    plt.show()
    
    
