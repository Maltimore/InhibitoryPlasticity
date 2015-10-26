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
simtime = 10000*ms         # Simulation time
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
sigma_c = 100             # connectivity spread
sigma_s = 100             # sensor width
fixed_in_degree = .02     # amount of incoming connections
start_weight = 10         # starting weight for the inh to exc connections
# a matrix to hold the inh to exc weights as they change over time
w_holder = np.zeros((int(NE * fixed_in_degree * NI),
                     int(simtime/rate_interval)))
w_holder[:, 0] = start_weight


# control parameters
do_plotting = True
do_global_update = False
do_local_update = True

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
connectivity_mat = mytools.create_connectivity_mat(
                                        sigma_c = sigma_c,
                                        N_pre = NE,
                                        N_post = NE,
                                        x_pre = 1,
                                        x_post = 1,
                                        fixed_in_degree = fixed_in_degree,
                                        save_to_file = True,
                                        reload_from_file = True,
                                        filename = "exc_to_exc",
                                        dir_name = "connectivity_matrices")
con_ee = Synapses(Pe, Pe, pre='g_ampa += 0.3*scaling_factor*nS')
con_ee.connect(connectivity_mat[:,0], connectivity_mat[:,1])

connectivity_mat = mytools.create_connectivity_mat(
                                        sigma_c = sigma_c,
                                        N_pre = NE,
                                        N_post = NI,
                                        x_pre = 1,
                                        x_post = 4,
                                        fixed_in_degree = fixed_in_degree,
                                        save_to_file = True,
                                        reload_from_file = True,
                                        filename = "exc_to_inh",
                                        dir_name = "connectivity_matrices")
con_ie = Synapses(Pe, Pi, pre='g_ampa += 0.3*scaling_factor*nS')
con_ie.connect(connectivity_mat[:,0], connectivity_mat[:,1])

connectivity_mat = mytools.create_connectivity_mat(
                                        sigma_c = sigma_c,
                                        N_pre = NI,
                                        N_post = NI,
                                        x_pre = 4,
                                        x_post = 4,
                                        fixed_in_degree = fixed_in_degree,
                                        save_to_file = True,
                                        reload_from_file = True,
                                        filename = "inh_to_inh",
                                        dir_name = "connectivity_matrices")
con_ii = Synapses(Pi, Pi, pre='g_gaba += 3*scaling_factor*nS')
con_ii.connect(connectivity_mat[:,0], connectivity_mat[:,1])

### PLASTIC SYNAPSES ##########################################################
print("Creating plastic synapses..")
ei_conn_mat = mytools.create_connectivity_mat(
                                        sigma_c = sigma_c,
                                        N_pre = NI,
                                        N_post = NE,
                                        x_pre = 4,
                                        x_post = 1,
                                        fixed_in_degree = fixed_in_degree,
                                        save_to_file = True,
                                        reload_from_file = True,
                                        filename = "inh_to_exc",
                                        dir_name = "connectivity_matrices")
con_ei = Synapses(Pi, Pe,
                  model='''w : 1
                           ''',
                  pre='''g_gaba += w*scaling_factor*nS
                         w += 1e-11                      
                         ''')
con_ei.connect(ei_conn_mat[:,0], ei_conn_mat[:,1])
con_ei.w = 10

### MONITORS ##################################################################
print("Setting up Monitors..")
SpikeMon = SpikeMonitor(neurons)
inhSpikeMon = SpikeMonitor(Pi)

### HELPER FUNCTIONS ##########################################################
def exponential_window(tau, dt):
    max_t = 5 * tau
    time = np.arange(0, max_t/ms, dt/ms) * ms
    window = 1 / tau * np.exp(-time/(tau))
    return time, window

### ARBITRARY PYTHON CODE #####################################################
@network_operation(dt=rate_interval)
def global_update(t):
    if t/ms == 0:
        # if this is t = 0, skip the computation
        return
    _, firing_rate = mytools.estimate_pop_firing_rate(inhSpikeMon, 
                         rate_interval, simtime,
                         t_min = t - rate_interval, t_max = t)
    con_ei.w += eta * (firing_rate - rho_0)

@network_operation(dt=rate_interval)
def local_update(t):
    if t/ms == 0:
        # if this is t = 0, skip the computation
        return
    _, firing_rates = mytools.estimate_single_firing_rates(inhSpikeMon, 
                         rate_interval, simtime,
                         t_min = t - rate_interval, t_max = t)
    firing_rates = mytools.rate_sensor(firing_rates, x_NI, sigma_s)
    
    temp_w_holder = np.array(con_ei.w)
    for neuron_idx in np.arange(NI):
        delta_w = eta * (firing_rates[neuron_idx] - rho_0)
        idxes = ei_conn_mat[:,0] == neuron_idx
        temp_w_holder[idxes] += delta_w
    con_ei.w = temp_w_holder
    w_holder[:, int(t/rate_interval)] = temp_w_holder



### NETWORK ###################################################################
print("Creating Network..")
MyNet = Network(neurons, Pe, Pi, con_ee, con_ie, con_ii, con_ei,
                SpikeMon, inhSpikeMon)
if do_global_update:
    MyNet.add(global_update)
if do_local_update:
    MyNet.add(local_update)
    
### SIMULATION ################################################################
print("Running simulation..")
MyNet.run(simtime, report="stdout")
print("Done simulating.")

### PLOTTING ##################################################################
if do_plotting:
    plot_script.create_plots(SpikeMon, inhSpikeMon, rate_interval, w_holder,
                             rho_0, simtime, dt)
else:
    print("Plotting was not desired.")

