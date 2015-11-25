from brian2 import *
import numpy as np
import plot_script
import mytools
import imp
import os
imp.reload(plot_script)
imp.reload(mytools)
start_scope()
set_device("cpp_standalone")

### PARAMETERS ################################################################
Ntot = 10000
NE = int(Ntot * 4/5)
NI = int(Ntot / 5)
scaling_factor = np.sqrt(10000 / Ntot)
x_NI = int(NE/NI)
w_ii =  3  * scaling_factor*nS

all_parameters = { \
    "Ntot": Ntot,
    "NE" :  NE                 , # Number of excitatory cells
    "NI" :  NI                 , # Number of inhibitory cells
    "x_NI" :  x_NI             , # spacing of inh cells
    "tau_ampa" : 5.0*ms        , # Glutamatergic synaptic time constant
    "tau_gaba" : 10.0*ms       , # GABAergic synaptic time constant
    "epsilon" : 0.02           , # Sparseness of synaptic connections
    "tau_stdp" : 20*ms         , # STDP time constant
    "rate_interval" : 1000*ms  , # bin size to compute firing rate
    "gl" : 10.0*nS             , # Leak conductance
    "el" : -60*mV              , # Resting potential
    "er" : -80*mV              , # Inhibitory reversal potential
    "vt" : -50.*mV             , # Spiking threshold
    "memc" : 200.0*pfarad      , # Membrane capacitance
    "bgcurrent" : 200*pA       , # External current
    "fixed_in_degree" : .02    , # amount of incoming connections
    "eta" : .05                , # Learning rate
    "rho_0" : 15               , # Target firing rate
    "scaling_factor" : scaling_factor,    
    "w_ee" : .3 * scaling_factor*nS,  	
    "w_ie" : .3  * scaling_factor*nS,	
    "w_ii" : w_ii,  	
    "wmin" : 0,
    "wmax" : w_ii * 100 / nS,
    "save_connectivity_to_file": True,
    "load_connectivity_from_file": True,
    
    
    
    "simtime" : 2001*ms        ,   # Simulation time
    "dt" : .1*ms               ,   # Simulation time step
    "plot_n_weights" : 200     ,   # Number of weights to be plotted
    "sigma_c" : 100            ,   # connectivity spread
    "sigma_s" : 50 / x_NI      ,   # sensor width adapted to spacing of inh cells
    "start_weight" : 8         ,   # starting weight for the inh to exc connections
    "do_plotting" : True       ,  
    "do_global_update" : False , 
    "do_local_update" : False  , 
    "do_profiling" : False     , 
    "do_run" : True            , 
    "program_dir" : os.getcwd()}
# extract variables from the dictionary to the global namespace
for key,val in all_parameters.items():
    exec(key + '=val')
    
np.random.seed(1337)
use_maltes_algorithm = False
use_owens_algorithm = not use_maltes_algorithm

### NEURONS ###################################################################
print("Creating neurons..")
eqs_neurons='''
dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er))+bgcurrent)/memc
        : volt (unless refractory)
dg_ampa/dt = -g_ampa/tau_ampa : siemens
dg_gaba/dt = -g_gaba/tau_gaba : siemens
A : 1
'''

neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                      reset="""v = el
                               A += 1""",
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
con_ee = Synapses(Pe, Pe, pre="""g_ampa += w_ee""")
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
con_ie = Synapses(Pe, Pi, pre='g_ampa += w_ie')
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
con_ii = Synapses(Pi, Pi, pre='g_gaba += w_ii')
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
con_ei.w = start_weight

### MONITORS ##################################################################
print("Setting up Monitors..")
k_in_ei = NI * epsilon
random_selection = np.floor(np.random.uniform(0, k_in_ei*NE, plot_n_weights))
inhWeightMon = StateMonitor(con_ei, "w", dt = rate_interval,
                            record=random_selection)

### RUN FUNCTION ##############################################################


def runnet2(sigma_s, NI, NE, rho_0, eta, wmin, wmax, rate_interval, simtime,
           network_objs, program_dir="/tmp/brian_source_files", run=True):
    def exponential_window(tau, dt):
        max_t = 5 * tau
        time = np.arange(0, max_t/ms, dt/ms) * ms
        window = 1 / tau * np.exp(-time/(tau))
        window /= np.sum(window)
        return time, window
    
    @network_operation(dt=rate_interval)
    def local_update(t):
        if t/ms == 0:
            # if this is t = 0, skip the computation
            return
        _, firing_rates = mytools.estimate_single_firing_rates(inhSpikeMon, 
                             rate_interval, simtime,
                             t_min = t - rate_interval, t_max = t)
        # save the computed firing rates for usage after the simulation ended
        # (the minus 1 is because at t = 0 we can't save anything)
        rate_holder[:, int(t/rate_interval)-1] = firing_rates
        
        # apply the rate sensor to the single firing rates
        firing_rates = mytools.rate_sensor(firing_rates, x_NI, sigma_s)
        
        temp_w_holder = np.array(con_ei.w)
        for neuron_idx in np.arange(NI):
            delta_w = eta * (firing_rates[neuron_idx] - rho_0)
            idxes = ei_conn_mat[:,0] == neuron_idx
            temp_w_holder[idxes] += delta_w
        # set below 0 weights to zero.
        temp_w_holder[temp_w_holder < 0] = 0
        con_ei.w = temp_w_holder
        # save the weights for later usage 
        w_holder[:, int(t/rate_interval)] = temp_w_holder
    
    inhSpikeMon = SpikeMonitor(network_objs["Pi"])    
    network_objs["local_update"] = local_update
    network_objs["inhSpikeMon"] = inhSpikeMon
    network_objs = list(set(network_objs.values()))
    net = Network(network_objs)
    net.run(simtime, report="stdout", profile=do_profiling)
              
### NETWORK ###################################################################
network_objs = {"neurons": neurons,
                    "Pe": Pe,
                    "Pi": Pi,
                    "con_ee": con_ee,
                    "con_ie": con_ie,
                    "con_ii": con_ii, 
                    "con_ei": con_ei,
                    "inhWeightMon": inhWeightMon}
    
### SIMULATION ################################################################
print("Starting runnet..")
if use_owens_algorithm:
    rateMon = mytools.run_cpp_standalone(all_parameters, network_objs)
elif use_maltes_algorithm:
    pass
print("Done simulating.")

### PLOTTING ##################################################################
if do_plotting:
    rate_holder = rateMon.A[:, 1:] / rate_interval
    w_holder = inhWeightMon.w
    plot_script.create_plots(False, False, rate_interval, rho_0, 
                             w_holder, rate_holder, simtime, dt)
else:
    print("Plotting was not desired.")
