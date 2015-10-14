from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
start_scope()

### PARAMETERS ################################################################
NE = 1000               # Number of excitatory cells
NI = NE/4               # Number of inhibitory cells
tau_ampa = 5.0*ms       # Glutamatergic synaptic time constant
tau_gaba = 10.0*ms      # GABAergic synaptic time constant
epsilon = 0.02          # Sparseness of synaptic connections
tau_stdp = 20*ms        # STDP time constant
simtime = 700*ms       # Simulation time
rate_interval = 200*ms  # bin size to compute firing rate

gl = 10.0*nS            # Leak conductance
el = -60*mV             # Resting potential
er = -80*mV             # Inhibitory reversal potential
vt = -50.*mV            # Spiking threshold
memc = 200.0*pfarad     # Membrane capacitance
bgcurrent = 200*pA      # External current

### VARIABLE DECLARATIONS #####################################################
firing_rate_list = []

### FUNCTIONS #################################################################
def get_n_of_spikes(spike_time_vec, t, intervallength):
    n_spikes = len(spike_time_vec[t-intervallength < spike_time_vec < t])
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
con_e = Synapses(Pe, neurons, pre='g_ampa += 0.3*nS', connect='rand()<epsilon')
con_ii = Synapses(Pi, Pi, pre='g_gaba += 3*nS', connect='rand()<epsilon')

### PLASTIC SYNAPSES ##########################################################
print("Creating plastic synapses..")
con_ei = Synapses(Pi, Pe,
                  model='''w : 1
                           pre_spikes_last_second : 1
                           pre_spikes_last_interval : 1
                           ''',
                  pre='''pre_spikes_last_second += 1
                         pre_spikes_last_interval += 1
                         g_gaba += w*nS
                         w += 1e-11                      
                         ''',
                  connect='rand()<epsilon')
con_ei.w = 300
con_ei.pre_spikes_last_second = 0
con_ei.pre_spikes_last_interval = 0
con_ei.run_regularly("""pre_spikes_last_second = 0""", dt=1000*ms)

### MONITORS ##################################################################
print("Setting up Monitors..")
StateMon = StateMonitor(con_ei, ['w', 'pre_spikes_last_second'], record=0)
SpikeMon = SpikeMonitor(neurons)
inh_SpikeMon = SpikeMonitor(Pi)
excStateMon = StateMonitor(Pe, "v", record=0)
inhStateMon = StateMonitor(Pi, "v", record=0)

### ARBITRARY PYTHON CODE #####################################################
@network_operation(dt=1000*ms)
def global_update():
    cum_sum = np.sum(con_ei.pre_spikes_last_second)
    print("cumulative sum is: " + str(cum_sum))
    con_ei.pre_spikes_last_second = 0 # reset counter
    
@network_operation(dt=rate_interval)
def compute_firing_rate(t):
    t = t/ms
    print("compute_firing_rate called at t = " + str(t))
    if t == 0:
        print("skipping first step")
        return
    firing_rate = np.sum(con_ei.pre_spikes_last_interval) \
    / rate_interval * ms / len(con_ei) * 1000
    con_ei.pre_spikes_last_interval = 0 # reset counter
    firing_rate_list.append([t, firing_rate])
    
### NETWORK ###################################################################
print("Creating Network..")
MyNet = Network(neurons, Pe, Pi, con_e, con_ii, con_ei, StateMon, inhStateMon,
                excStateMon, SpikeMon, global_update, compute_firing_rate)
    
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

show_n = 10
plt.figure()
plt.plot(SpikeMon.t/ms, SpikeMon.i, '.k')
plt.ylim([NE, NE+10])
plt.xlabel("Time (ms)")
plt.ylabel("Neuron index")
plt.title("Spikes in the first " + str(show_n) + " inhibitory neurons")


# synaptic weight
plt.figure()
plt.plot(StateMon.t/ms, StateMon.w.T)
plt.xlabel("time [ms]")
plt.ylabel("weight")
plt.title("Synaptic weight of inhibitory synapse #1")

plt.figure()
plt.plot(StateMon.t/ms, StateMon.pre_spikes_last_second.T)
plt.xlabel("time [ms]")
plt.ylabel("# Spikes")
plt.title("Spikes this bin in the presynaptic neuron of inhibitory synapse #1")
plt.xlim([-10, np.amax(StateMon.t/ms)])
plt.ylim([0, np.amax(StateMon.pre_spikes_last_second)+1])

plt.figure()
plt.plot(inhStateMon.t/ms, inhStateMon.v.T/volt, label="inh #1")
plt.plot(excStateMon.t/ms, excStateMon.v.T/volt, label="exc #1")
plt.xlabel("time [ms]")
plt.ylabel("Voltage")
plt.legend()
plt.title("Voltage traces")

plt.figure()
plt.plot(firing_rate_list[0,:], firing_rate_list[1,:])
#plt.ylim([0, np.amax(firing_rate_list[1,:])])
plt.xlim([0, firing_rate_list[0,-1]])
plt.xlabel("time [ms]")
plt.ylabel("firing rate [Hz]")
plt.title("Average inhibitory firing rate over time")
