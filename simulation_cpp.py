print("Python script is running!", flush=True)
from brian2 import *
import numpy as np
import mytools
import imp
import os
import sys
import pickle
imp.reload(mytools)
start_scope()

### PARAMETERS ################################################################
neuron_scaling = 2
Ntot = int(10000 / neuron_scaling)
NE = int(Ntot * 4/5)
NI = int(Ntot / 5)
scaling_f = np.sqrt(neuron_scaling)
x_NI = int(NE/NI)


params = { \
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
    "scaling_f" : scaling_f,    
    "w_ee" : .3 * scaling_f*nS,  	
    "w_ie" : .3  * scaling_f*nS,	
    "w_ii" : 3  * scaling_f*nS,  	
    "w_ei" : 3 * scaling_f,   # starting weight for the inh to exc connections
    "wmin" : float(0),
    "wmax" : float(300),
    "save_connectivity_to_file": True,
    "load_connectivity_from_file": True,
    
    
    "prep_time" : 20000*second    ,   # give Network time to stabilize
    "simtime" :  300.001*second,   # Simulation time
    "dt" : .1*ms               ,   # Simulation time step
    "sigma_c" : 200            ,   # connectivity spread
    "sigma_s" : 200            ,   # sensor width
    "do_plotting" : False      ,  
    "do_global_update" : False , 
    "do_local_update" : False  , 
    "do_profiling" : False     , 
    "do_run" : True            , 
    "program_dir" : os.getcwd()}
# extract variables from the dictionary to the global namespace
for key,val in params.items():
    exec(key + '=val')

if __name__ == "__main__":
    user_params = mytools.parse_argvs(sys.argv, neuron_scaling)
    if user_params == "invalid":
        print("User input was invalid. Exiting..")
        sys.stdout.flush()
        sys.exit(0)
    elif user_params == "parameter_file_requested":
        print("Parameter file requested.")         
    else:
        params["sigma_s"] = user_params[0]
        params["sigma_c"] = user_params[1]

# adding parameters to be saved
resultspath = program_dir + "/results/rho0_" + rho_0 + "Hz/"
results = {}
results["prep_time"] = params["prep_time"]
results["simtime"] = params["simtime"]
results["rho_0"] = params["rho_0"]
results["w_min"] = params["w_min"]
results["w_max"] = params["w_max"]
results["eta"] = params["eta"]
results["lookuptable"] = mytools.lookuptable(neuron_scaling)
         

if user_params == "parameter_file_requested":
    print("Saving parameter file")
    if not os.path.exists(resultspath + "rates_and_weights"):
        os.makedirs(resultspath + "rates_and_weights")
    pickle.dump(results, open(resultspath + "rates_and_weights/"
                              + "parameter_file", "wb"))
    sys.stdout.flush()
    sys.exit(0)

print("I'm running a simulation with sigma_s = " + str(sigma_s) + \
      " and sigma_c = " + str(sigma_c))

np.random.seed(1337)
use_maltes_algorithm = False
use_owens_algorithm = not use_maltes_algorithm
params["cpp_standalone"] = use_owens_algorithm
if params["cpp_standalone"]:
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
Pe = neurons[:NE]
Pi = neurons[NE:]
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
                  model='''w : 1''',
                  pre='''g_gaba += w*scaling_f*nS
                         w += 1e-11
                      ''',
                         name="con_ei")
con_ei.connect(ei_conn_mat[:,0], ei_conn_mat[:,1])
con_ei.w = w_ei
params["ei_conn_mat"] = ei_conn_mat # saving this particular conn.
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
    mytools.run_cpp_standalone(params, network_objs)
elif use_maltes_algorithm:
    mytools.run_old_algorithm(params, network_objs)
print("Done simulating.")


### SAVE RESULTS ##############################################################
### When saving values to disk, we are not taking into account Brian units.
### Threfore we are assigning standard units so that when recovering the 
### saved files, one knows which units to assign to them.
### time: second
### weights: nS
resultfile = "sigma_s_" + str(params["sigma_s"]) + "_" + \
             "sigma_c_" + str(params["sigma_c"]) + "_" + \
             "prep_" + str(int(params["prep_time"]/second)) + "_seconds"

# adding data to be saved
results["inhWeights"] = network_objs["inhWeightMon"].w # no unit actually!
results["weight_times"] = network_objs["inhWeightMon"].t/second
results["inh_rates"] = network_objs["rateMon"].A / (params["rate_interval"] / second)
results["inh_rate_times"] = network_objs["rateMon"].t/second
results["all_inh_weights"] = network_objs["con_ei"].w[:]

if not os.path.exists(resultspath + "rates_and_weights"):
    os.makedirs(resultspath + "rates_and_weights")
pickle.dump(results, open(resultspath + "rates_and_weights/"
                          + resultfile, "wb"))
results["inh_spike_times"] = network_objs["inhSpikeMon"].t/second
results["inh_spike_neuron_idxes"] = network_objs["inhSpikeMon"].i[:]
results["exc_spike_times"] = network_objs["excSpikeMon"].t/second
results["exc_spike_neuron_idxes"] = network_objs["excSpikeMon"].i[:]
if not os.path.exists(resultspath + "all_data"):
    os.makedirs(resultspath + "all_data")
pickle.dump(results, open(resultspath + "all_data/"
                          + resultfile, "wb"))