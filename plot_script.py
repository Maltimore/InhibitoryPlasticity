from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import mytools
import imp
imp.reload(mytools)

    
def create_plots(SpikeMon, inhSpikeMon, rate_interval, wMon, rho_0,
                 simtime, dt):
    print("Creating plots..")
    N_inh_neurons = len(inhSpikeMon.spike_trains())    
    # all spikes
    plt.figure()
    plt.plot(SpikeMon.t/ms, SpikeMon.i, '.k', markersize=.1)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")

    # first show_n inhibitory neurons spikes
    show_n = 20
    plt.figure()
    plt.plot(inhSpikeMon.t/ms, inhSpikeMon.i, '.k')
    plt.ylim([0, show_n])
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.title("Spikes in the first " + str(show_n) + " inhibitory neurons")
    
    # inhibitory firing rate over time    
    times, rates = mytools.estimate_pop_firing_rate(inhSpikeMon, rate_interval,
                                                    simtime)
    plt.figure()
    plt.plot(times, rates, color="blue", label="firing_rate")
    plt.hlines(rho_0, times[0], times[-1], linestyles="--")
    plt.xlabel("time [ms]")
    plt.ylabel("firing rate [Hz]")
    plt.twinx()
    plt.plot(times, wMon.w[0], color="red", label="weight")
    plt.ylabel("Inh to exc weight")
    plt.ylim([0, max(wMon.w[0])+2])
    plt.title("Firing rates estimated every " + str(rate_interval))
        
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