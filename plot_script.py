from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

def create_plots(SpikeMon, inhSpikeMon, excStateMon, inhStateMon,
                 rateMon, dt):
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
    
    tau = 100 * ms
    firing_rate = filters.gaussian_filter1d(rateMon.rate/Hz,
                                            tau/dt, mode='reflect')
    plt.figure()
    plt.plot(rateMon.t/ms, firing_rate)
    plt.xlabel("time [ms]")
    plt.ylabel("firing rate [Hz]")
    plt.title("Filtered with gaussian kernel (SD = " + str(tau) + ")")