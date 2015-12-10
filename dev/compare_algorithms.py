from brian2 import *
import numpy as np
import mytools_compare_algorithms as mytools
import imp
import os
import sys
import pickle
imp.reload(mytools)
start_scope()


use_maltes_algorithm = False
use_cpp = False
use_owens_algorithm = not use_maltes_algorithm
if use_maltes_algorithm:
    # in this case, override my earlier decision to use cpp because it was
    # clearly nonsense (I don't have cpp code generation)
    use_cpp = False
if use_owens_algorithm:
    if not use_cpp:
        prefs.codegen.target = 'numpy'
        

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
    "eta" : .01                , # Learning rate
    "rho_0" : 15               , # Target firing rate
    "scaling_factor" : scaling_factor,    
    "w_ee" : .3 * scaling_factor*nS,  	
    "w_ie" : .3  * scaling_factor*nS,	
    "w_ii" : w_ii,  	
    "wmin" : float(0),
    "wmax" : float(w_ii * 100 / nS),
    "save_connectivity_to_file": True,
    "load_connectivity_from_file": True,
    
    
    "prep_time" : 0*second    ,   # give Network time to stabilize
    "simtime" :  1005*ms      ,   # Simulation time
    "dt" : .1*ms               ,   # Simulation time step
    "plot_n_weights" : 200     ,   # Number of weights to be plotted
    "sigma_c" : 200            ,   # connectivity spread
    "sigma_s" : 100     ,   # sensor width adapted to spacing of inh cells
    "start_weight" : 8         ,   # starting weight for the inh to exc connections
    "do_plotting" : False      ,  
    "do_global_update" : False , 
    "do_local_update" : False  , 
    "do_profiling" : False     , 
    "do_run" : True            , 
    "program_dir" : os.getcwd()}

if __name__ == "__main__":
    user_params = mytools.parse_argvs(sys.argv)
    if user_params != "invalid":
        all_parameters["sigma_s"] = user_params[0]
        all_parameters["sigma_c"] = user_params[1]
    else:
        print("User input was invalid.")

# extract variables from the dictionary to the global namespace
for key,val in all_parameters.items():
    exec(key + '=val')
    
np.random.seed(1337)

all_parameters["cpp_standalone"] = use_cpp
if all_parameters["cpp_standalone"]:
    set_device("cpp_standalone")

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
Pi = neurons[:NI]
Pe = neurons[NI:]

neurons.v = np.random.uniform(el, vt-2*mV, len(neurons))*volt 

### SYNAPSES ##################################################################
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
con_ee = Synapses(Pe, Pe, pre="""g_ampa += w_ee""", name="con_ee")
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
con_ie = Synapses(Pe, Pi, pre='g_ampa += w_ie', name="con_ie")
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
con_ii = Synapses(Pi, Pi, pre='g_gaba += w_ii', name="con_ii")
con_ii.connect(connectivity_mat[:,0], connectivity_mat[:,1])

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
                         ''',
                         name="con_ei")
con_ei.connect(ei_conn_mat[:,0], ei_conn_mat[:,1])
con_ei.w = start_weight
all_parameters["ei_conn_mat"] = ei_conn_mat # saving this particular conn.
                                            # matrix for later use

### MONITORS ##################################################################
print("Setting up Monitors..")
k_in_ei = NI * epsilon
random_selection = np.floor(np.random.uniform(0, k_in_ei*NE, plot_n_weights))
inhWeightMon = StateMonitor(con_ei, "w", dt = rate_interval,
                            record=random_selection)
rateMon = StateMonitor(Pi, "A", record=True,
                       when="start",
                       dt=rate_interval)
inhSpikeMon = SpikeMonitor(Pi)
excSpikeMon = SpikeMonitor(Pe)
              
### NETWORK ###################################################################
monitors =     {"inhWeightMon": inhWeightMon,
                "rateMon": rateMon,
                "inhSpikeMon": inhSpikeMon,
                "excSpikeMon": excSpikeMon}
                
network_objs = {"neurons": neurons,
                "Pe": Pe,
                "Pi": Pi,
                "con_ee": con_ee,
                "con_ie": con_ie,
                "con_ii": con_ii, 
                "con_ei": con_ei,
                "monitors": monitors}
    
### SIMULATION ################################################################
print("Starting run function..")
if use_owens_algorithm:
    mytools.run_cpp_standalone(all_parameters, network_objs)
elif use_maltes_algorithm:
    mytools.run_old_algorithm(all_parameters, network_objs)
print("Done simulating.")


### SAVE RESULTS ##############################################################
### When saving values to disk, we are not taking into account Brian units.
### Threfore we are assigning standard units so that when recovering the 
### saved files, one knows which units to assign to them.
### time: second
### weights: nS
r_hat = network_objs["r_hat_mon"].r_hat[:]
resultspath = program_dir + "/results"
if not os.path.exists(resultspath):
    os.makedirs(resultspath)

results = {}
results["inhWeights"] = network_objs["inhWeightMon"].w # no unit actually!
results["weight_times"] = network_objs["inhWeightMon"].t/second
results["inh_rates"] = network_objs["rateMon"].A
results["inh_rate_times"] = network_objs["rateMon"].t/second
results["prep_time"] = all_parameters["prep_time"]
results["inh_spike_times"] = network_objs["inhSpikeMon"].t/second
results["inh_spike_neuron_idxes"] = network_objs["inhSpikeMon"].i[:]
results["exc_spike_times"] = network_objs["excSpikeMon"].t/second
results["exc_spike_neuron_idxes"] = network_objs["excSpikeMon"].i[:]
results["r_hat"] = network_objs["r_hat_mon"].r_hat[:]
results["sigma_s"] = all_parameters["sigma_s"]
results["NI"] = all_parameters["NI"]
results["x_NI"] = all_parameters["x_NI"]
results["kernel_used"] = all_parameters["kernel_to_export"]
if not os.path.exists(resultspath + "/comparison"):
    os.makedirs(resultspath + "/comparison")
if use_maltes_algorithm:
    pickle.dump(results, open(resultspath + "/comparison/"
                              + "malte_results.p", "wb"))
elif use_cpp:
    pickle.dump(results, open(resultspath + "/comparison/"
                              + "owen_results_cpp.p", "wb"))
else:
    pickle.dump(results, open(resultspath + "/comparison/"
                              + "owen_results_numpy.p", "wb"))
    

                          
### PLOTTING ##################################################################
if do_plotting:
    import plot_script
    imp.reload(plot_script) # this is just for development purposes
    rate_holder = network_objs["rateMon"].A[:, 1:] / rate_interval
    w_holder = network_objs["inhWeightMon"].w
    plot_script.create_plots(all_parameters, w_holder, rate_holder)
else:
    print("Plotting was not desired.")
