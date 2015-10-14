from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
start_scope()

### PARAMETERS ################################################################
Ntot = 10000
NE = int(Ntot * 4/5)  # Number of excitatory cells
NI = int(Ntot / 5)      # Number of inhibitory cells
tau_ampa = 5.0*ms       # Glutamatergic synaptic time constant
tau_gaba = 10.0*ms      # GABAergic synaptic time constant
epsilon = 0.02          # Sparseness of synaptic connections
tau_stdp = 20*ms        # STDP time constant
simtime = 10000*ms       # Simulation time
rate_interval = 200*ms  # bin size to compute firing rate
gl = 10.0*nS            # Leak conductance
el = -60*mV             # Resting potential
er = -80*mV             # Inhibitory reversal potential
vt = -50.*mV            # Spiking threshold
memc = 200.0*pfarad     # Membrane capacitance
bgcurrent = 200*pA      # External current
scaling_factor = np.sqrt(10000 / Ntot)
eta = .05

### VARIABLE DECLARATIONS #####################################################
firing_rate_list = []

### FUNCTIONS #################################################################
def get_n_of_spikes(spike_time_vec, t, intervallength):
    n_spikes = len(spike_time_vec[(t-intervallength < spike_time_vec) & \
                                  (spike_time_vec < t)])
    return n_spikes
### NEURONS ###################################################################
print("Creating neurons..")
eqs_neurons='''
dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er))+bgcurrent)/memc
        : volt (unless refractory)
dg_ampa/dt = -g_ampa/tau_ampa : siemens
dg_gaba/dt = -g_gaba/tau_gaba : siemens
'''

neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                      reset="""v=el""",
                      refractory=5*ms)
Pe = neurons[:NE]
Pi = neurons[NE:]
neurons.v = np.random.uniform(el, vt, len(neurons))*volt 

### NONPLASTIC SYNAPSES #######################################################
print("Creating nonplastic synapses..")
con_e = Synapses(Pe, neurons, pre='g_ampa += 0.3*scaling_factor*nS',
                 connect='rand()<epsilon')
con_ii = Synapses(Pi, Pi, pre='g_gaba += 3*scaling_factor*nS',
                  connect='rand()<epsilon')

### PLASTIC SYNAPSES ##########################################################
print("Creating plastic synapses..")
con_ei = Synapses(Pi, Pe,
                  model='''w : 1
                           ''',
                  pre='''g_gaba += w*scaling_factor*nS
                         w += 1e-11                      
                         ''',
                  connect='rand()<epsilon')
con_ei.w = 3

### MONITORS ##################################################################
print("Setting up Monitors..")
#StateMon = StateMonitor(con_ei, ['w'], record=0)
SpikeMon = SpikeMonitor(neurons)
inhSpikeMon = SpikeMonitor(Pi)
excStateMon = StateMonitor(Pe, "v", record=0)
inhStateMon = StateMonitor(Pi, "v", record=0)

### ARBITRARY PYTHON CODE #####################################################
@network_operation(dt=rate_interval)
def compute_inh_firing_rate(t):
    t = t/ms
    spike_times = inhSpikeMon.t/ms
    if t == 0:
        # if this is t = 0, skip the computation
        return
    firing_rate = get_n_of_spikes(spike_times, t, rate_interval/ms)/     \
                  (rate_interval / second * NI)
    firing_rate_list.append([t, firing_rate])
    con_ei.w += eta * (firing_rate - 15)
    
### NETWORK ###################################################################
print("Creating Network..")
MyNet = Network(neurons, Pe, Pi, con_e, con_ii, con_ei, inhStateMon,
                excStateMon, SpikeMon, compute_inh_firing_rate,
                inhSpikeMon)
    
### SIMULATION ################################################################
print("Running simulation..")
MyNet.run(simtime, report="stdout")

### PLOTTING ##################################################################
# make python list to np array (first row is time, second row is firing rate)
firing_rate_list = np.array(firing_rate_list).T

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

# Firing rate
plt.figure()
plt.plot(firing_rate_list[0,:], firing_rate_list[1,:])
plt.xlim([0, firing_rate_list[0,-1]])
plt.xlabel("time [ms]")
plt.ylabel("firing rate [Hz]")
plt.title("Average inhibitory firing rate over time")
plt.show()

