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
rate_interval = 1000*ms    # bin size to compute firing rate
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
plot_n_weights = 200
sigma_c = 100             # connectivity spread
sigma_s = 50 / x_NI       # sensor width adapted to spacing of inh cells
start_weight = 8          # starting weight for the inh to exc connections
do_plotting = False
do_global_update = False
do_local_update = False
do_profiling = False
do_run = True
mypath = "cpp_standalone"
np.random.seed(1337)

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
def runnet(sigma_s, NI, NE, rho_0, eta, wmin, wmax, rate_interval,
           network_objs, tempdir="/tmp/brian_source_files", run=True):
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
    neurons = network_objs["neurons"]
    con = con_ei
    rho0_dt = rho_0/second * rate_interval

    # Includes the header files in all generated files
    prefs.codegen.cpp.headers += ['<sense.h>',]
    prefs.codegen.cpp.define_macros += [('N_REAL', int(N)),
                                        ('N_CMPLX', int(N/2+1))]
    path_to_sense_hpp = os.path.join(tempdir, 'sense.h') 
    path_to_sense_cpp = os.path.join(tempdir, 'sense.cpp')
    with open(path_to_sense_hpp, "w") as f:
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
    with open(path_to_sense_cpp, "w") as f:
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
    if sigma_s == np.infty:
        k = np.ones(N)/N
    elif sigma_s < 1e-3: 
        k = np.zeros(N)
        k[0] = 1
    else:
        intercell = 1.
        length = intercell*N
        d = np.linspace(intercell-length/2, length/2, N)
        d = np.roll(d, int(N/2+1))
        k = np.exp(-np.abs(d)/sigma_s)
        k /= k.sum()
    rate_vars =  '''k : 1
                    r_hat : 1
                    r_hat_single : 1'''
    kg = NeuronGroup(N, model=rate_vars, name='kernel_rates')
    kg.active = False
    kg.k = k #kernel in the spatial domain
    network_objs["dummygroup"] = kg

    rateMon = StateMonitor(network_objs["Pi"], "A", record=True,
                           when="start",
                           dt=rate_interval)
    network_objs["rateMon"] = rateMon
    
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
    '''.format(A=device_get_array_name(neurons.variables['A']),
               r_hat=device_get_array_name(kg.variables['r_hat']),
               Noffset=Noffset)
    @implementation('cpp', custom_code)
    @check_units(_=Unit(1), result=Unit(1), discard_units=True)
    def spatial_filter(_):
        kg.r_hat = irfft(K * rfft(neurons.A), N).real
        neurons.A = 0
        return 0
    neurons.run_regularly('dummy = spatial_filter()',
                          dt=rate_interval, order=1,
                          name='filterspatial')


    custom_code = '''
    double update_weights(double w, int32_t i_pre)
    {{
        w += {eta}*(brian::{r_hat}[i_pre] - {rho0_dt});
        return std::max({wmin}, std::min(w, {wmax}));
    }}
    '''.format(r_hat=device_get_array_name(kg.variables['r_hat']), eta=eta,
               rho0_dt = rho0_dt, wmin='0.0', wmax=wmax)

    @implementation('cpp', custom_code)
    @check_units(w=Unit(1), i_pre=Unit(1), result=Unit(1), discard_units=True)
    def update_weights(w, i_pre):
        del_W = eta*(kg.r_hat - rho0_dt)
        w += del_W[i_pre]
        np.clip(w, wmin, wmax, out=w)
        return w
    con.run_regularly('w = update_weights(w, i)', dt=rate_interval, when='end',  name='weightupdate' )
    # i is the presynaptic index (brian knows this automatically, j would be postsynaptic)
        
    network_objs = list(set(network_objs.values()))
    net = Network(network_objs)
    
    if run:
        print("Starting cpp standalone compilation..")    
        net.run(simtime, report='text')
        additional_source_files = [path_to_sense_cpp,]
        build = CurrentDeviceProxy.__getattr__(device, 'build')
        build(directory=tempdir, compile=True, run=True, debug=False, 
              additional_source_files=additional_source_files)
        return rateMon
    else:
        return 0
              
              
### NETWORK ###################################################################
MyNetworkObjects = {"neurons": neurons,
                    "Pe": Pe,
                    "Pi": Pi,
                    "con_ee": con_ee,
                    "con_ie": con_ie,
                    "con_ii": con_ii, 
                    "con_ei": con_ei,
                    "inhWeightMon": inhWeightMon}
    
### SIMULATION ################################################################
print("Starting runnet..")
rateMon = runnet(sigma_s = sigma_s,
               NI=NI,
               NE=NE,
               rho_0=rho_0,
               eta=eta,
               wmin=wmin,
               wmax=wmax,
               rate_interval = rate_interval,
               network_objs = MyNetworkObjects,
               tempdir = mypath,
               run = do_run)
print("Done simulating.")

### PLOTTING ##################################################################
if do_plotting:
    rate_holder = rateMon.A[:, 1:] / rate_interval
    w_holder = inhWeightMon.w
    plot_script.create_plots(False, False, rate_interval, rho_0, 
                             w_holder, rate_holder, simtime, dt)
else:
    print("Plotting was not desired.")
