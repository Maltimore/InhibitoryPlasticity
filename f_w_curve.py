from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import plot_script
import mytools
import imp
imp.reload(plot_script)
imp.reload(mytools)
start_scope()

### PARAMETERS ################################################################
Ntot = 10000
NE = int(Ntot * 4/5)      # Number of excitatory cells
NI = int(Ntot / 5)        # Number of inhibitory cells
tau_ampa = 5.0*ms         # Glutamatergic synaptic time constant
tau_gaba = 10.0*ms        # GABAergic synaptic time constant
epsilon = 0.02            # Sparseness of synaptic connections
tau_stdp = 20*ms          # STDP time constant
simtime = 1000*ms         # Simulation time
dt = .1*ms                # Simulation time step
rate_interval = 500*ms    # bin size to compute firing rate
gl = 10.0*nS              # Leak conductance
el = -60*mV               # Resting potential
er = -80*mV               # Inhibitory reversal potential
vt = -50.*mV              # Spiking threshold
memc = 200.0*pfarad       # Membrane capacitance
bgcurrent = 200*pA        # External current
eta = .05                 # Learning rate
rho_0 = 15                # Target firing rate
scaling_factor = np.sqrt(10000 / Ntot)
# control parameters
do_plotting = False
do_global_update = False

stepsize = .1
weights = np.arange(0, 10, stepsize)
rate_vec = np.zeros(len(weights))
for idx, current_weight in enumerate(weights):
    ### NEURONS ###############################################################
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
    
    ### NONPLASTIC SYNAPSES ###################################################
    print("Creating nonplastic synapses..")
    con_e = Synapses(Pe, neurons, pre='g_ampa += 0.3*scaling_factor*nS',
                     connect='rand()<epsilon')
    con_ii = Synapses(Pi, Pi, pre='g_gaba += 3*scaling_factor*nS',
                      connect='rand()<epsilon')
    
    ### PLASTIC SYNAPSES ######################################################
    print("Creating plastic synapses..")
    con_ei = Synapses(Pi, Pe,
                      model='''w : 1
                               ''',
                      pre='''g_gaba += w*scaling_factor*nS
                             w += 1e-11                      
                             ''',
                      connect='rand()<epsilon')
    con_ei.w = current_weight
    
    ### MONITORS ##############################################################
    print("Setting up Monitors..")
    SpikeMon = SpikeMonitor(neurons)
    inhSpikeMon = SpikeMonitor(Pi)
    wMon = StateMonitor(con_ei, variables="w", record=0, dt=rate_interval)
    
    ### HELPER FUNCTIONS ######################################################
    def exponential_window(tau, dt):
        max_t = 5 * tau
        time = np.arange(0, max_t/ms, dt/ms) * ms
        window = 1 / tau * np.exp(-time/(tau))
        return time, window
    
    ### ARBITRARY PYTHON CODE #################################################
    @network_operation(dt=rate_interval)
    def global_update(t):
        if t/ms == 0:
            # if this is t = 0, skip the computation
            return
        # extract spike times
        spike_train = inhSpikeMon.t/ms
        # cut out spike times further in the past than the time of the window
        spike_train = spike_train[spike_train > (t - rate_interval)/ms]
        # estimate firing rate by dividing number of spikes by window time
        firing_rate = len(spike_train) / (rate_interval/second * NI)
        con_ei.w += eta * (firing_rate - rho_0)
        return
    
    ### NETWORK ###############################################################
    print("Creating Network..")
    MyNet = Network(neurons, Pe, Pi, con_e, con_ii, con_ei, SpikeMon,
                    inhSpikeMon, wMon)
    if do_global_update:
        MyNet.add(global_update)
        
    ### SIMULATION ############################################################
    print("Running simulation..")
    MyNet.run(simtime, report="stdout")
    print("Done simulating.")
    
    _, firing_rate = mytools.estimate_pop_firing_rate(
                         inhSpikeMon, rate_interval, simtime, 
                         t_min = simtime - rate_interval)
    print("Estimated firing rate for weight " + str(current_weight) + \
          " was " + str(firing_rate))
    rate_vec[idx] = firing_rate
plt.figure()
plt.plot(weights, rate_vec)
plt.xlabel("inh to exc weight [nS]")
plt.ylabel("firing rate [Hz]")
plt.title("Inhibitory population average rate, w stepsize: " + str(stepsize))
plt.show()
