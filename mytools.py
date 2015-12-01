from brian2 import *
import numpy as np
import pickle
import os

### HELPER FUNCTIONS FOR TOOL FUNCTIONS #######################################
def _find_nearest(array,value):
    distances = (np.abs(array-value))
    idx = np.argmin(distances)
    return idx
    
def _exp_function(x_vec, mu, scale):
    # catch the case where scale == 0
    if scale == 0:
        y_vec = np.zeros(len(x_vec))
        y_vec[x_vec==mu] = 1
        return y_vec
    elif scale == "ininity":
        # if an infinitely big sensor is desired, return uniform weights
        y_vec = np.ones(len(x_vec))
        y_vec /= np.sum(y_vec)
        return y_vec
    #else, compute normal exponential function
    return 1/scale * np.exp(2* -np.abs(x_vec - mu) / scale)

### TOOL FUNCTIONS ############################################################
def parse_argvs(argv):
    n_sensors = 13
    n_connectivities = 5
    n_total = n_sensors * n_connectivities
    
    sensor_widths = list(2 ** np.arange(12))
    sensor_widths[0] = 0
    sensor_widths.append("infinity")
    sensor_widths = [val for val in sensor_widths for _ in range(n_connectivities)]
    
    connectivities = [100, 300, 600, 1000, "infinity"] * n_sensors
    
    all_args = list(zip(sensor_widths, connectivities))
    
    
    
    if len(argv) < 2:
        print("Call for simulation didn't receive enough arguments.")
        print("Exiting..")
        return "invalid"
    elif len(argv) > 2:
        print("Call for simulation received too many arguments.")
        print("Exiting..")
        return "invalid"
    elif int(argv[1]) > n_total:
        # notice that qsub indexes from 1 therefor it's not ">="
        print("Index " + str(argv[1]) + " doesn't match any parameter set.")
        print("Exiting..")
        return "invalid"
    else:
        index = int(argv[1]) - 1 # qsub cannot give task ID 0.
        sigma_s = all_args[index][0]
        sigma_c = all_args[index][1]
        
        print("I'm running a simulation with sigma_s = " + str(sigma_s) + \
              " and sigma_c = " + str(sigma_c))
    return sigma_s, sigma_c


def rate_sensor(firing_rates, x_NI, sigma_s):
    """ Compute the firing rates per neuron with an exponential window across
        the neighboring neurons. The exponential window is paraemtrized by
        sigma_s, which is the width of the exponential. """
    N_neurons = len(firing_rates)
    # Creating the exponential window and set its maximum over the "middle"
    # (in the vector) neuron. Later we will just take this window and rotate
    # it according to our needs (according to which neurons firing rate we're
    # estimating).
    mu = int(N_neurons/2) * x_NI
    x_vec = np.arange(N_neurons) * x_NI
    y_vec = _exp_function(x_vec, mu, sigma_s)
    y_vec /= np.sum(y_vec)
    
    sensor_rates = np.zeros(N_neurons)
    for neuron_idx in np.arange(N_neurons):
        y_vec_temp = np.roll(y_vec, int(neuron_idx - (mu/x_NI)))
        
        sensor_rates[neuron_idx] = np.dot(y_vec_temp, firing_rates)
    
    return sensor_rates


def estimate_pop_firing_rate(SpikeMon, rate_interval, simtime, t_min = 0*ms,
                             t_max = "end of sim"):
    """If t_max - t_min < rate_interval, will still use an entire rate
       interval to compute firing rate (i.e., then t_min is discarded in the
       sense that still an entire interval will be computed, hower just one)"""
    if t_max == "end of sim":
        t_max = simtime
    N_neurons = len(SpikeMon.spike_trains())
    spike_times = SpikeMon.t
    upper_bound_times = np.arange((t_min + rate_interval)/ms, t_max/ms + 1,
                                  rate_interval/ms) * ms
    rate_vec = np.zeros(len(upper_bound_times))
    for idx, upper_bound in enumerate(upper_bound_times):
        mask = (upper_bound - rate_interval < spike_times) & \
               (spike_times < upper_bound)
        n_spikes = len(spike_times[mask])
        rate_vec[idx] = n_spikes / (rate_interval/second * N_neurons)

    # in case the firing rates were computed for just one time interval, delete
    # the superfluous axis    
    rate_vec = rate_vec.squeeze()
    return upper_bound_times, rate_vec

def estimate_single_firing_rates(SpikeMon, rate_interval, simtime,
                                 t_min = "t_max - rate_interval",
                                 t_max = "end of sim",
                                 N_neurons = "all"):
    if t_max == "end of sim":
        t_max = simtime
    if t_min == "t_max - rate_interval":
        t_min = t_max - rate_interval
    if N_neurons == "all":
        N_neurons = len(SpikeMon.spike_trains())
    
    spike_times = SpikeMon.t
    upper_bound_times = np.arange((t_min + rate_interval)/ms, t_max/ms + 1,
                                  rate_interval/ms) * ms
    rate_mat = np.zeros((N_neurons, len(upper_bound_times)))
    for time_idx, upper_bound in enumerate(upper_bound_times):
        for neuron_idx in np.arange(N_neurons):
            # extract spike times of just one neuron
            single_spike_times = spike_times[SpikeMon.i == neuron_idx]
            mask = (upper_bound - rate_interval < single_spike_times) & \
                   (single_spike_times < upper_bound)
            single_spike_times = single_spike_times[mask]
            # estimate firing rate by dividing number of spikes by window time
            firing_rate = len(single_spike_times) / (rate_interval/second)
            # save firing rate in matrix
            rate_mat[neuron_idx, time_idx] = firing_rate
  
    # in case the firing rates were computed for just one time interval, delete
    # the superfluous axis
    rate_mat = rate_mat.squeeze()
    return upper_bound_times, rate_mat

def _create_connectivity_filename(raw_filename, sigma_c, N_pre, N_post):
    """ Creates a long filename that uniquely identifies a connectivity
        matrix file.
    """
    new_filename = raw_filename + "__sigma_c_" + str(sigma_c) + "__N_pre_" + \
                   str(N_pre) + "__N_post_" + str(N_post)
    return new_filename

def create_connectivity_mat(sigma_c = 500,
                            N_pre = 8000,
                            N_post = 2000,
                            x_pre = 1,
                            x_post = 4,
                            fixed_in_degree = 0.02,
                            save_to_file = True,
                            reload_from_file = True,
                            filename = "no_name_specified",
                            dir_name = "connectivity_matrices"):
    full_filename = _create_connectivity_filename(filename, sigma_c,
                                                  N_pre, N_post)
    if reload_from_file:
        if os.path.exists(dir_name + "/" + full_filename):
            print("Loading connecitivity matrix from file " \
                  + full_filename + "..", end="", flush=True)                        
            conn_mat = pickle.load(open(dir_name + "/" + full_filename, "rb"))            
            print(" Done.", flush=True)            
            return conn_mat
    print("Couldn't find connectivity matrix on disk. Creating..", end="",
          flush=True)
    pre_idxes = np.arange(N_pre)
    post_idxes = np.arange(N_post)
    pre_positions = pre_idxes * x_pre
    
    k_in = int(fixed_in_degree * N_pre)
    all_pre_neurons = np.zeros(k_in * N_post, dtype=int)
    all_post_neurons = np.zeros(k_in * N_post, dtype=int)
    
    for post_idx in post_idxes:
        # holder variable for the pre neurons for the current post neuron
        pre_neurons = np.ones(k_in, dtype=int) * -1
        
        for pre_idx in np.arange(k_in):
            # set flag that the chosen neuron will be unacceptable.
            # flag will be set to true once it passed our checks.        
            chosen_pre_unacceptable = True
            
            while chosen_pre_unacceptable:
                # draw from an exponential distribution
                # (add 1 so no self projections)
                rand_pre_pos = np.random.exponential(scale=sigma_c) + 1
                rand_pre_pos *= np.random.randint(0, 2)*2 - 1
                rand_pre_pos += post_idx * x_post
                
                while rand_pre_pos > N_pre * x_pre:
                    rand_pre_pos -= N_pre * x_pre
                while rand_pre_pos * x_pre < 0:
                    rand_pre_pos += N_pre * x_pre
                pre_neuron = _find_nearest(pre_positions, rand_pre_pos)
                
                if pre_neuron not in pre_neurons:
                    pre_neurons[pre_idx] = pre_neuron                
                    chosen_pre_unacceptable = False
                    
    
        all_pre_neurons[k_in*post_idx:k_in*(post_idx+1)] = pre_neurons
        all_post_neurons[k_in*post_idx:k_in*(post_idx+1)] = \
            np.ones(k_in) * post_idx
    
    connectivity_mat = np.vstack((all_pre_neurons, all_post_neurons)).T
    print("Done", flush=True)
    
    if save_to_file:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.exists(dir_name + "/" + full_filename):
            print("Saving connecitivity matrix to file " \
                  + full_filename + "..", flush=True) 
            pickle.dump(connectivity_mat,
                        open(dir_name + "/" + full_filename, "wb"))
    
    return connectivity_mat


def compute_sparseness(rate_holder):
    rates = rate_holder[:,-1]    
    NI = len(rates)
    sparseness = ((np.sum(rates)/NI)**2) / (np.sum(rates**2)/NI)
    return sparseness
    
    
def run_cpp_standalone(params, network_objs):
    import os
    from numpy.fft import rfft, irfft
    from brian2.devices.device import CurrentDeviceProxy
    from brian2.units import Unit
    from brian2 import check_units, implementation, device, prefs, NeuronGroup, Network

    tempdir = os.path.join(params["program_dir"], 'cpp_standalone')
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
                  
                  
    prefs.codegen.cpp.libraries += ['mkl_gf_lp64', # -Wl,--start-group
                                    'mkl_gnu_thread',
                                    'mkl_core', #  -Wl,--end-group
                                    'iomp5']

    
    # give extra arguments and path information to the compiler    
    extra_incs = ['-I'+os.path.expanduser(s) for s in [ tempdir, "~/intel/mkl/include"]]
    prefs.codegen.cpp.extra_compile_args_gcc = ['-w', '-Ofast', '-march=native'] + extra_incs

    # give extra arguments and path information to the linker
    prefs.codegen.cpp.extra_link_args += ['-L{0}/intel/mkl/lib/intel64'.format(os.path.expanduser('~')),
                                          '-L{0}/intel/lib/intel64'.format(os.path.expanduser('~')),
                                          '-m64', '-Wl,--no-as-needed']
    
    # Path that the compiled and linked code needs at runtime
    os.environ["LD_LIBRARY_PATH"] = os.path.expanduser('~/intel/mkl/lib/intel64:')
    os.environ["LD_LIBRARY_PATH"] += os.path.expanduser('~/intel/lib/intel64:')
                                       
    # Variable definitions
    N = params["NI"] # this is the amount of neurons with variable synaptic strength
    Noffset = params["NE"]
    neurons = network_objs["neurons"]
    params["rho0_dt"] = params["rho_0"]/second * params["rate_interval"]
    mkl_threads = 1


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
    if params["sigma_s"] == np.infty:
        k = np.ones(N)/N
    elif params["sigma_s"] < 1e-3: 
        k = np.zeros(N)
        k[0] = 1
    else:
        intercell = params["x_NI"]
        length = intercell*N
        d = np.linspace(intercell-length/2, length/2, N)
        d = np.roll(d, int(N/2+1))
        k = np.exp(-np.abs(d)/params["sigma_s"])
        k /= k.sum()
    rate_vars =  '''k : 1
                    r_hat : 1
                    r_hat_single : 1'''
    kg = NeuronGroup(N, model=rate_vars, name='kernel_rates')
    kg.active = False
    kg.k = k #kernel in the spatial domain
    network_objs["dummygroup"] = kg


    
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
    network_objs["neurons"].run_regularly('dummy = spatial_filter()',
                          dt=params["rate_interval"], order=1,
                          name='filterspatial')
    params["spatial_filter"] = spatial_filter

    custom_code = '''
    double update_weights(double w, int32_t i_pre)
    {{
        w += {eta}*(brian::{r_hat}[i_pre] - {rho0_dt});
        return std::max({wmin}, std::min(w, {wmax}));
    }}
    '''.format(r_hat=device_get_array_name(kg.variables['r_hat']), 
               eta=params["eta"], rho0_dt = params["rho0_dt"], 
               wmin=params["wmin"], wmax=params["wmax"])
    

    @implementation('cpp', custom_code)
    @check_units(w=Unit(1), i_pre=Unit(1), result=Unit(1), discard_units=True)
    def update_weights(w, i_pre):
        del_W = params["eta"]*(kg.r_hat - rho0_dt)
        w += del_W[i_pre]
        np.clip(w, params["wmin"], params["wmax"], out=w)
        return w
    network_objs["con_ei"].run_regularly('w = update_weights(w, i)',
                                         dt=params["rate_interval"],
                                         when='end', name='weightupdate')
    # i is the presynaptic index (brian
    # knows this automatically, j would be postsynaptic)
    params["update_weights"] = update_weights
    
    
    temp_objs = network_objs.copy()
    temp_objs.pop("inhWeightMon")
    temp_objs.pop("rateMon")
    net = Network(list(set(temp_objs.values())))
    
    if not params["do_run"]:
        print("Running the network was not desired")
        return

    if params["prep_time"]/second > 0:
        print("Starting prep time run")
        net.run(params["prep_time"], report='text', namespace = params)
        additional_source_files = [path_to_sense_cpp,]
        build = CurrentDeviceProxy.__getattr__(device, 'build')
        build(directory=tempdir, compile=True, run=True, debug=False, 
              additional_source_files=additional_source_files)
        print("Prep time run complete.", flush=True)
    
    # Add the Monitors only now so we don't record unnecessarily much.
    net.add(network_objs["inhWeightMon"])
    net.add(network_objs["rateMon"])        

    print("Starting real simulation")
    net.run(params["simtime"], report='text', namespace = params)
    additional_source_files = [path_to_sense_cpp,]
    build = CurrentDeviceProxy.__getattr__(device, 'build')
    build(directory=tempdir, compile=True, run=True, debug=False, 
          additional_source_files=additional_source_files)
    


def run_old_algorithm(params, network_objs):
    @network_operation(dt=params["rate_interval"], order=1)
    def local_update(t):
        if t/ms == 0:
            # if this is t = 0, skip the computation
            return
        firing_rates = network_objs["Pi"].A / (params["rate_interval"] / second)
        network_objs["Pi"].A = 0
        
        # apply the rate sensor to the single firing rates
        firing_rates = rate_sensor(firing_rates, 
                                   params["x_NI"],
                                   params["sigma_s"])
        
        temp_w_holder = np.array(network_objs["con_ei"].w)
        for neuron_idx in np.arange(params["NI"]):
            delta_w = params["eta"] * (firing_rates[neuron_idx] - params["rho_0"])
            idxes = params["ei_conn_mat"][:,0] == neuron_idx
            temp_w_holder[idxes] += delta_w
        # set below 0 weights to zero.
        temp_w_holder = clip(temp_w_holder, params["wmin"], params["wmax"])
        network_objs["con_ei"].w = temp_w_holder
    
    network_objs["local_update"] = local_update
    net = Network(list(set(network_objs.values())))
    net.run(params["simtime"], report="stdout",
            profile= params["do_profiling"], namespace = params)
