from brian2 import *
import numpy as np
import pickle
import os

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
    
def _find_nearest(array,value):
    distances = (np.abs(array-value))
    idx = np.argmin(distances)
    return idx
    
def create_connectivity_mat(sigma_c = 500,
                            N_pre = 8000,
                            N_post = 2000,
                            x_pre = 1,
                            x_post = 4,
                            fixed_in_degree = 0.02,
                            save_to_file = False,
                            reload_from_file = True,
                            filename = "no_name_specified",
                            dir_name = "connectivity_matrices"):
    
    if reload_from_file:
        if os.path.exists(dir_name + "/" + filename):
            print("Loading connecitivity matrix from file " \
                  + filename + "..", end="", flush=True)                        
            conn_mat = pickle.load(open(dir_name + "/" + filename, "rb"))            
            print(" Done.", flush=True)            
            return conn_mat

    pre_idxes = np.arange(N_pre)
    post_idxes = np.arange(N_post)
    pre_positions = pre_idxes * x_pre
    
    k_in = int(fixed_in_degree * N_pre)
    pre_neurons = np.zeros(k_in * N_post)
    post_neurons = np.zeros(k_in * N_post)
    
    for post_idx in post_idxes:
        # draw from an exponential distribution (add 1 so no self projections)
        rand_pre_positions = np.random.exponential(scale=sigma_c, size=k_in) + 1
        rand_pre_positions *= np.random.randint(0, 2, size=k_in)*2 - 1
        rand_pre_positions += post_idx * x_post
        pre_neurons_helper = np.zeros(k_in)
        for pos_idx, curr_pre_pos in enumerate(rand_pre_positions):
            while curr_pre_pos > N_pre * x_pre:
                curr_pre_pos -= N_pre * x_pre
            while curr_pre_pos * x_pre < 0:
                curr_pre_pos += N_pre * x_pre
            pre_neurons_helper[pos_idx] = _find_nearest(pre_positions,
                                                        curr_pre_pos)
        pre_neurons[k_in*post_idx:k_in*(post_idx+1)] = pre_neurons_helper
        post_neurons[k_in*post_idx:k_in*(post_idx+1)] = \
            np.ones(k_in) * post_idx
    connectivity_mat = np.vstack((pre_neurons, post_neurons)).astype(int).T
    
    if save_to_file:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.exists(dir_name + "/" + filename):
            print("Saving connecitivity matrix to file " \
                  + filename + "..", flush=True) 
            pickle.dump(connectivity_mat,
                        open(dir_name + "/" + filename, "wb" ))
    
    return connectivity_mat






