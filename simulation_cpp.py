from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import plot_script
import mytools
import imp
imp.reload(plot_script)
imp.reload(mytools)
start_scope()
set_device("cpp_standalone")

### PARAMETERS ################################################################
Ntot = 10000
NE = int(Ntot * 4/5)      # Number of excitatory cells
NI = int(Ntot / 5)        # Number of inhibitory cells
x_NI = int(NE/NI)         # spacing of inh cells
tau_ampa = 5.0*ms         # Glutamatergic synaptic time constant
tau_gaba = 10.0*ms        # GABAergic synaptic time constant
epsilon = 0.02            # Sparseness of synaptic connections
tau_stdp = 20*ms          # STDP time constant
rate_interval = 500*ms    # bin size to compute firing rate
gl = 10.0*nS              # Leak conductance
el = -60*mV               # Resting potential
er = -80*mV               # Inhibitory reversal potential
vt = -50.*mV              # Spiking threshold
memc = 200.0*pfarad       # Membrane capacitance
bgcurrent = 200*pA        # External current
fixed_in_degree = .02     # amount of incoming connections
eta = .05                 # Learning rate
rho_0 = 15                # Target firing rate
scaling_factor = np.sqrt(10000 / Ntot)
w_ee = .3 * scaling_factor*nS
w_ie = .3  * scaling_factor*nS
w_ii = 3  * scaling_factor*nS
wmin = 0
wmax = w_ii * 100 / nS

### Parameters I'm actually changing ##########################################
simtime = 20000*ms        # Simulation time
simtime += 1*ms           # adding one so that things like np.arange create
                          # a bin for the last interval
dt = .1*ms                # Simulation time step
sigma_c = 100             # connectivity spread
sigma_s = 0            # sensor width
start_weight = 8          # starting weight for the inh to exc connections
do_plotting = True
do_global_update = False
do_local_update = False
do_profiling = False
mypath = "/home/malte/Dropbox/Studium/Lab_Rotation_1/program/cpp_standalone"

### VARIABLE DECLARATIONS #####################################################
# a matrix to hold the inh to exc weights as they change over time
# notice that i'm adding one for the number of columns because of the
# "fencepost error" (wikipedia it)
w_holder = np.zeros((int(NE * fixed_in_degree * NI),
                     int(simtime/rate_interval)+1))
w_holder[:, 0] = start_weight
rate_holder = np.zeros((NI, int(simtime/rate_interval)))

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


### RUN FUNCTION ##############################################################
def runnet(sigma_c, NI, NE, rho_0, eta, wmin, wmax, rate_interval,
           network_objs, tempdir="/tmp/brian_source_files"):
    import os
    from numpy.fft import rfft, irfft
    from brian2.devices.device import CurrentDeviceProxy
    from brian2.units import Unit
    from brian2 import check_units, implementation, device, prefs, NeuronGroup, Network

    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
                  
                  
    prefs.codegen.cpp.libraries += ['mkl_gf_lp64', # -Wl,--start-group
                                    'mkl_gnu_thread',
                                    'mkl_core', #  -Wl,--end-group
                                    'iomp5']    
    prefs.codegen.cpp.extra_link_args += ['-L{0}/miniconda3/lib'.format(os.path.expanduser('~')),
                                          '-m64', '-Wl,--no-as-needed']

    
    os.environ["LD_LIBRARY_PATH"] = os.path.expanduser('~/miniconda3/lib:')   ## for the linker 
    extra_incs = ['-I'+os.path.expanduser(s) for s in [ tempdir, "~/intel/mkl/include"]]
    prefs.codegen.cpp.extra_compile_args_gcc = ['-w', '-Ofast', '-march=native'] + extra_incs
    mkl_threads = 1
    
    # Variable definitions
    N = NI # this is the amount of neurons with variable synaptic strength
    Noffset = NE
    pop = neurons
    con = con_ei
    wdt = rate_interval
    rho0_dt = rho_0/second * wdt

    # Includes the header files in all generated files
    prefs.codegen.cpp.headers += ['<sense.h>',]
    prefs.codegen.cpp.define_macros += [('N_REAL', int(N)),
                                        ('N_CMPLX', int(N/2+1))]
    sense_hpp = os.path.join(tempdir, 'sense.h') 
    sense_cpp = os.path.join(tempdir, 'sense.cpp')
    with open(sense_hpp, "w") as f:
        header_code = '''
        #ifndef SENSE_H
        #define SENSE_H
        #include <mkl_service.h>
        #include <mkl_vml.h>
        #include <mkl_dfti.h>
        #include <cstring>
        extern DFTI_DESCRIPTOR_HANDLE hand;
        extern MKL_Complex16 in_cmplx[N_CMPLX], out_cmplx[N_CMPLX], k_cmplx[N_CMPLX];
        DFTI_DESCRIPTOR_HANDLE init_dfti();
        #endif'''
        f.write(header_code)
        #MKL_Complex16 is a type (probably struct)
    with open(sense_cpp, "w") as f:
        sense_code = '''
        #include <sense.h>
        DFTI_DESCRIPTOR_HANDLE hand;
        MKL_Complex16 in_cmplx[N_CMPLX], out_cmplx[N_CMPLX], k_cmplx[N_CMPLX];
        DFTI_DESCRIPTOR_HANDLE init_dfti()
        {{
            DFTI_DESCRIPTOR_HANDLE hand = 0;
            mkl_set_num_threads({mkl_threads});
            DftiCreateDescriptor(&hand, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N_REAL); //MKL_LONG status
            DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            DftiSetValue(hand, DFTI_BACKWARD_SCALE, 1. / N_REAL);
            //if (0 == status) status = DftiSetValue(hand, DFTI_THREAD_LIMIT, {mkl_threads});
            DftiCommitDescriptor(hand); //if (0 != status) cout << "ERROR, status = " << status << "\\n";
            return hand;
        }} '''.format(mkl_threads=mkl_threads, )
        f.write(sense_code)

    # device_get_array_name will be the function get_array_name() and what it does is getting
    # the string names of brian objects
    device_get_array_name = CurrentDeviceProxy.__getattr__(device, 'get_array_name')   
    # instert_code is a function which is used to insert code into the main()
    # function
    insert_code = CurrentDeviceProxy.__getattr__(device, 'insert_code')
   
    ### Computing the kernel (Owen changed it to a gaussian kernel now)
    # Owen uses a trick here which is he creates a NeuronGroup which doesn't
    # really do anything in the Simulation. It's just a dummy NeuronGroup
    # to hold an array to which he would like to have access to during runtime.   
    if sigma_c == np.infty:
        k = np.ones(N)/N
    elif sigma_c < 1e-3: 
        k = np.zeros(N)
        k[0] = 1
    else:
        intercell = 1.
        length = intercell*N
        d = np.linspace(intercell-length/2, length/2, N)
        d = np.roll(d, int(N/2+1))
        k = np.exp(-np.abs(d)/sigma_c)
        k /= k.sum()
    rate_vars =  '''k : 1
                    r_hat : 1'''
    kg = NeuronGroup(N, model=rate_vars, name='kernel_rates')
    kg.active = False
    kg.k = k #kernel in the spatial domain
    network_objs.append(kg)

    main_code = '''
    hand = init_dfti(); 
    DftiComputeForward(hand, brian::{k}, k_cmplx); 
    '''.format(k=device_get_array_name(kg.variables['k']))
    insert_code('main', main_code) # DftiComp.. writes its result into k_cmplx
    K = rfft(k)
    
    # Variable A is a spike counter
    # memset resets the array to zero (memset is defined to take voidpointers)
    # also the star before *brian tells it to not compute the size of the 
    # pointer, but what the pointer points to
    # the _num_ thing is that whenever there's an array in brian,
    # it automatically creates an integer of the same name with _num_ 
    # in front of it (and that is the size)
    custom_code = '''
    double spatial_filter(int)
    {{
        DftiComputeForward(hand, brian::{A}+{Noffset}, in_cmplx);
        vzMul(N_CMPLX, in_cmplx, k_cmplx, out_cmplx);
        DftiComputeBackward(hand, out_cmplx, brian::{r_hat});
        memset(brian::{A}, 0, brian::_num_{A}*sizeof(*brian::{A}));
        return 0;
    }}
    '''.format(A=device_get_array_name(pop.variables['A']), r_hat=device_get_array_name(kg.variables['r_hat']), Noffset=Noffset)
    @implementation('cpp', custom_code)
    @check_units(_=Unit(1), result=Unit(1), discard_units=True)
    def spatial_filter(_):
        kg.r_hat = irfft(K * rfft(pop.A), N).real
        pop.A = 0
        return 0
    pop.run_regularly('dummy = spatial_filter()', dt=wdt, when='start', name='filterspatial')


    custom_code = '''
    double update_weights(double w, int32_t i_pre)
    {{
        w += {eta}*(brian::{r_hat}[i_pre] - {rho0_dt});
        return std::max({wmin}, std::min(w, {wmax}));
    }}
    '''.format(r_hat=device_get_array_name(kg.variables['r_hat']), eta=eta, rho0_dt=rho0_dt, wmin='0.0', wmax=wmax)

    @implementation('cpp', custom_code)
    @check_units(w=Unit(1), i_pre=Unit(1), result=Unit(1), discard_units=True)
    def update_weights(w, i_pre):
        del_W = eta*(kg.r_hat - rho0_dt)
        w += del_W[i_pre]
        np.clip(w, wmin, wmax, out=w)
        return w
    con.run_regularly('w = update_weights(w, i)', dt=wdt, when='end',  name='weightupdate' )
    # i is the presynaptic index (brian knows this automatically, j would be postsynaptic)
        
#    network_objs = list(set(simulation_objects.values()))
    net = Network(network_objs)

    net.run(simtime, report='text')
    
    path_to_sense_cpp = os.path.join(tempdir, 'sense.cpp')
    additional_source_files = [path_to_sense_cpp,]
    build = CurrentDeviceProxy.__getattr__(device, 'build')
    build(directory=tempdir, compile=True, run=True, debug=False, 
          additional_source_files=additional_source_files)
              
              
              
              
              
### NETWORK ###################################################################
print("Creating Network..")
#MyNet = Network(neurons, Pe, Pi, con_ee, con_ie, con_ii, con_ei,
#                SpikeMon, inhSpikeMon)
MyNetworkObjects = [neurons, Pe, Pi, con_ee, con_ie, con_ii, con_ei,
                    SpikeMon, inhSpikeMon]
    
### SIMULATION ################################################################
print("Starting cpp standalone compilation..")
runnet(sigma_c = sigma_c,
       NI=NI,
       NE=NE,
       rho_0=rho_0,
       eta=eta,
       wmin=wmin,
       wmax=wmax,
       rate_interval = rate_interval,
       network_objs = MyNetworkObjects,
       tempdir = mypath)
print("Done simulating.")

### PLOTTING ##################################################################
if do_plotting:
    plot_script.create_plots(SpikeMon, inhSpikeMon, rate_interval, rho_0, 
                             w_holder, rate_holder, simtime, dt)
else:
    print("Plotting was not desired.")
