from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import plot_script
import imp
imp.reload(plot_script)
import scipy.ndimage.filters as filters
start_scope()

### PARAMETERS ################################################################
Ntot = 10000
NE = int(Ntot * 4/5)    # Number of excitatory cells
NI = int(Ntot / 5)      # Number of inhibitory cells
tau_ampa = 5.0*ms       # Glutamatergic synaptic time constant
tau_gaba = 10.0*ms      # GABAergic synaptic time constant
epsilon = 0.02          # Sparseness of synaptic connections
tau_stdp = 20*ms        # STDP time constant
simtime = 5000*ms       # Simulation time
dt = .1*ms              # Simulation time step
rate_interval = 200*ms  # bin size to compute firing rate
gl = 10.0*nS            # Leak conductance
el = -60*mV             # Resting potential
er = -80*mV             # Inhibitory reversal potential
vt = -50.*mV            # Spiking threshold
memc = 200.0*pfarad     # Membrane capacitance
bgcurrent = 200*pA      # External current
eta = .1                # Learning rate
rho_0 = 15              # Target firing rate
tau_smoothing = 100*ms  # SD for gaussian smoothing kernel of firing rate
scaling_factor = np.sqrt(10000 / Ntot)

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
con_ei.w = 10

### MONITORS ##################################################################
print("Setting up Monitors..")
#StateMon = StateMonitor(con_ei, ['w'], record=0)
SpikeMon = SpikeMonitor(neurons)
inhSpikeMon = SpikeMonitor(Pi)
excStateMon = StateMonitor(Pe, "v", record=0)
inhStateMon = StateMonitor(Pi, "v", record=0)
rateMon = PopulationRateMonitor(Pi)

### ARBITRARY PYTHON CODE #####################################################
@network_operation(dt=rate_interval)
def compute_inh_firing_rate(t):
    t = t/ms
    if t == 0:
        # if this is t = 0, skip the computation
        return
    time = rateMon.t / ms
    timemask = time > (t - 1000)    
    rateMonrate = rateMon.rate / Hz
    rateMonrate = rateMonrate[timemask] # cut out rates of last 1000 ms only    
    firing_rate = filters.gaussian_filter1d(rateMonrate,
                                            tau_smoothing/dt, mode="reflect")
    firing_rate = np.average(firing_rate[-10:])
    con_ei.w += eta * (firing_rate - rho_0)
    print("Time is: " + str(t) + " ms")    
    print("The firing rate was: " + str(firing_rate))
    print("Delta w ist: " + str(eta*(firing_rate - rho_0)))
    print("")
    
### NETWORK ###################################################################
print("Creating Network..")
MyNet = Network(neurons, Pe, Pi, con_e, con_ii, con_ei, inhStateMon,
                excStateMon, SpikeMon, compute_inh_firing_rate,
                inhSpikeMon, rateMon)
    
### SIMULATION ################################################################
print("Running simulation..")
MyNet.run(simtime, report="stdout")
print("Done simulating.")
### PLOTTING ##################################################################
plot_script.create_plots(SpikeMon, inhSpikeMon, excStateMon, inhStateMon,
                         rateMon, dt)


