print("Python script is running!", flush=True)
from brian2 import *
import numpy as np
import mytools
import imp
import os
import sys
import pickle
import signal
imp.reload(mytools)
start_scope()

print("My pid is: " + str(os.getpid()), flush=True) 
sys.stdout.flush()
def signal_term_handler(signal, frame):
    print("Process with index " + str(sys.argv[1]) + " caught kill signal!")
    errorpath = os.getcwd() + "/errors"
    if not os.path.exists(errorpath):
        os.makedirs(errorpath)
    with open(errorpath + "/Error_in_qsub_index_" + str(sys.argv[1]) + \
              ".txt", "w") as text_file:
        print("This process caught kill signal.", file=text_file)
    sys.exit(0)
signal.signal(signal.SIGTERM, signal_term_handler)

### PARAMETERS #################################################################
neuron_scaling = 2 # this determines Ntot, Ntot is 10000 divided by this number
Ntot = int(10000 / neuron_scaling)
scaling_f = np.sqrt(neuron_scaling)


normal_mode = True # normal mode is if NE and NI are in the 'biological' config
if normal_mode:
    NE = 4000
    NI = 1000
    x_NE = 1
    x_NI = 4
    initial_exc_w = float(.3 * scaling_f)
    initial_inh_w = float( 3 * scaling_f)
else:
    NE = 1000
    NI = 4000
    x_NE = 4
    x_NI = 1
    initial_exc_w = float(.908 * scaling_f)
    initial_inh_w = float(.567 * scaling_f)


params = { \
    "Ntot": Ntot               , # Total number of neurons
    "NE" :  NE                 , # Number of excitatory cells
    "NI" :  NI                 , # Number of inhibitory cells
    "x_NE" :  x_NE             , # spacing of exc cells    
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
    "exc_bg_current" : 120*pA  , # External current
    "inh_bg_current" : 120*pA  , # External current
    "fixed_in_degree" : .02    , # amount of incoming connections
    "eta" : .01                , # Learning rate
    "rho_0" : 15                , # Target firing rate
    "scaling_f" : scaling_f    , # scaling factor if not using 10000 neurons   
    "w_ee" : initial_exc_w*nS  , # exc-exc weight
    "w_ie" : initial_exc_w*nS  , # exc-inh weight
    "w_ii" : initial_inh_w*nS  , # inh-inh weight
    "w_ei" : initial_inh_w     , # starting weight for the inh to exc connections
    "wmin" : float(0)          , # minimum permissible weight
    "wmax" : 100*initial_inh_w , # maximum permissible weight
    "prep_time" : 20000*second  , # give Network time to stabilize
    "simtime" :  300.001*second, # Simulation time
    "dt" : .1*ms               , # Simulation time step
    "sigma_c" : 200            , # connectivity spread
    "sigma_s" : 200            , # sensor width
    "plot_n_weights": 200      , # number of weights that will be recorded
    "do_profiling" : False     , # whether Brian should profile comp. times
    "do_run" : True            , # whether the simulation should actually run
    "program_dir" : os.getcwd(), # working directory
    "save_connectivity_to_file": True,   # whether to load connectivity matrix
    "load_connectivity_from_file": True,
    "simulation_name": "default_name"} #whether to save connectivity matrix


if __name__ == "__main__":
    user_params = mytools.parse_argvs(sys.argv, neuron_scaling)
    if user_params[0] == "parameter_file_requested":
        params["simulation_name"] = user_params[2]
        print("Parameter file requested.")         
    else:
        params["sigma_s"] = user_params[0]
        params["sigma_c"] = user_params[1]
        params["simulation_name"] = user_params[2]
        print("Normal operations started for the simulation with the name"
              + params["simulation_name"])

# extract variables from the dictionary to the global namespace
for key,val in params.items():
    exec(key + '=val')
    
# adding parameters to be saved
resultspath = program_dir + "/results/fewresults" + "/" + simulation_name + \
              "_rho0_" + str(rho_0) + "Hz/"
allresultspath = program_dir + "/results/allresults" + "/" + simulation_name + \
              "_rho0_" + str(rho_0) + "Hz/"
results = {}
results["NE"] = params["NE"]
results["NI"] = params["NI"]
results["x_NI"] = params["x_NI"]
results["x_NE"] = params["x_NE"]
results["initial_exc_weight"] = initial_exc_w
results["initial_inh_weight"] = initial_inh_w
results["exc_bg_current"] = params["exc_bg_current"]
results["inh_bg_current"] = params["inh_bg_current"]
results["prep_time"] = params["prep_time"]
results["simtime"] = params["simtime"]
results["rho_0"] = params["rho_0"]
results["wmin"] = params["wmin"]
results["wmax"] = params["wmax"]
results["eta"] = params["eta"]
results["lookuptable"] = mytools.lookuptable(neuron_scaling)
results["neuron_scaling"] = neuron_scaling
         
if user_params[0] == "parameter_file_requested":
    print("Saving parameter file")
    if not os.path.exists(resultspath):
        os.makedirs(resultspath)
    pickle.dump(results, open(resultspath + "parameter_file", "wb"))
    sys.stdout.flush()
    sys.exit(0)

print("I'm running a simulation with sigma_s = " + str(sigma_s) + \
      " and sigma_c = " + str(sigma_c), flush=True)

#np.random.seed(1337)
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
bgcurrent: amp
'''

neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                      reset="""v = el
                               A += 1""",
                      refractory=5*ms)
Pe = neurons[:NE]
Pi = neurons[NE:]
Pe.bgcurrent = exc_bg_current
Pi.bgcurrent = inh_bg_current
neurons.v = np.random.uniform(el, vt-2*mV, len(neurons))*volt 

### SYNAPSES ##################################################################
print("Creating nonplastic synapses..")
# E to E connections
connectivity_mat = mytools.create_connectivity_mat(
                                        sigma_c = sigma_c,
                                        N_pre = NE,
                                        N_post = NE,
                                        x_pre = x_NE,
                                        x_post = x_NE,
                                        fixed_in_degree = fixed_in_degree,
                                        save_to_file = True,
                                        reload_from_file = True,
                                        filename = "exc_to_exc",
                                        dir_name = "connectivity_matrices")
con_ee = Synapses(Pe, Pe, pre="""g_ampa += w_ee""", name="con_ee")
con_ee.connect(connectivity_mat[:,0], connectivity_mat[:,1])

# E to I connections
connectivity_mat = mytools.create_connectivity_mat(
                                        sigma_c = sigma_c,
                                        N_pre = NE,
                                        N_post = NI,
                                        x_pre = x_NE,
                                        x_post = x_NI,
                                        fixed_in_degree = fixed_in_degree,
                                        save_to_file = True,
                                        reload_from_file = True,
                                        filename = "exc_to_inh",
                                        dir_name = "connectivity_matrices")
con_ie = Synapses(Pe, Pi, pre='g_ampa += w_ie', name="con_ie")
con_ie.connect(connectivity_mat[:,0], connectivity_mat[:,1])

# I to I connections
connectivity_mat = mytools.create_connectivity_mat(
                                        sigma_c = sigma_c,
                                        N_pre = NI,
                                        N_post = NI,
                                        x_pre = x_NI,
                                        x_post = x_NI,
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
                                        x_pre = x_NI,
                                        x_post = x_NE,
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

if not os.path.exists(resultspath):
    os.makedirs(resultspath)
pickle.dump(results, open(resultspath + "/" + resultfile, "wb"))
                          

results["inh_spike_times"] = network_objs["inhSpikeMon"].t/second
results["inh_spike_neuron_idxes"] = network_objs["inhSpikeMon"].i[:]
results["exc_spike_times"] = network_objs["excSpikeMon"].t/second
results["exc_spike_neuron_idxes"] = network_objs["excSpikeMon"].i[:]
if not os.path.exists(allresultspath):
    os.makedirs(allresultspath)
pickle.dump(results, open(allresultspath + "/" + resultfile, "wb"))
