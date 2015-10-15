from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

def create_plots(SpikeMon, inhSpikeMon, excStateMon, inhStateMon,
                 firing_rate_list):
    print("Creating plots..")
    # spikes
    plt.figure()
    plt.plot(SpikeMon.t/ms, SpikeMon.i, '.k')
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    
    #inhibitory spikes
    show_n = 20
    plt.figure()
    plt.plot(inhSpikeMon.t/ms, inhSpikeMon.i, '.k')
    plt.ylim([0, show_n])
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.title("Spikes in the first " + str(show_n) + " inhibitory neurons")
    
    # Voltage traces
    plt.figure()
    plt.plot(inhStateMon.t/ms, inhStateMon.v.T/volt, label="inh #1")
    plt.plot(excStateMon.t/ms, excStateMon.v.T/volt, label="exc #1")
    plt.xlabel("time [ms]")
    plt.ylabel("Voltage")
    plt.legend()
    plt.title("Voltage traces")
    
    compute_firing_rate(SpikeMon.t)
    # Firing rate
    plt.figure()
    plt.plot(firing_rate_list[0,:], firing_rate_list[1,:])
    plt.xlim([0, firing_rate_list[0,-1]])
    plt.xlabel("time [ms]")
    plt.ylabel("firing rate [Hz]")
    plt.title("Average inhibitory firing rate over time")
    plt.show()
    
def compute_firing_rate(spikevector):
    # convert spike times to integer
    pass