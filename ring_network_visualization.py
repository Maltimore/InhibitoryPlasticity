import numpy as np
import matplotlib.pyplot as plt
import os

def _find_nearest(array,value):
    distances = (np.abs(array-value))
    idx = np.argmin(distances)
    return int(idx)

def exp_function(x_vec, mu, scale):
    return np.exp(-np.abs(x_vec - mu) / scale)

fixed_in_degree = 0.02
sigma_c = 500
N_pre = 8000
N_post = 8000
x_pre = 1
x_post = 1
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


# checking that there are no two synapses from the same presynaptic neuron
all_unique = True
for post_neuron_idx in post_idxes:
    pre_idxes = all_pre_neurons[post_neuron_idx*k_in : (post_neuron_idx+1)*k_in]
    unique_elements = np.unique(pre_idxes)
    if not len(unique_elements) == k_in:
        print("More than one synapse from same presynaptic neuron found.")
        all_unique = False        
        break
if all_unique:
    print("No neuron neuron hast two synapses from the same " \
          "presynaptic neuron")
    
    
    
# checking whether the algorithm works by visualizing the connectivity
n_bins = 8000
bin_edges = np.linspace(0, N_pre, n_bins+1)
hist_mat = np.zeros((N_post, n_bins))
for post_neuron_idx in post_idxes:
    pre_idxes = all_pre_neurons[post_neuron_idx*k_in : (post_neuron_idx+1)*k_in]
    hist, _ = np.histogram(pre_idxes, bins=bin_edges)
    hist_mat[post_neuron_idx, :] = hist

plt.figure()
plt.pcolormesh(hist_mat)
plt.xlabel("pre bins")
plt.ylabel("post neuron index")
plt.title("Connectivity hist colormap")

# visualizing the overall histogram
if True:
    idx_vec = np.zeros(N_pre*N_post)
    n_bins = N_pre
    bin_edges = np.linspace(0, N_pre, n_bins+1)
    overall_hist = np.zeros(n_bins)
    for post_neuron_idx in post_idxes:
        pre_idxes = all_pre_neurons[post_neuron_idx*k_in : \
                                    (post_neuron_idx+1)*k_in]
        hist, _ = np.histogram(pre_idxes, bins=bin_edges)
        hist = np.roll(hist, int(n_bins/2)-int(post_neuron_idx*x_post/x_pre))
        idx_vec[post_neuron_idx*N_pre:(post_neuron_idx+1)*N_pre] = hist        
        overall_hist += hist
    
    true_function = exp_function(np.arange(8000), 4000, sigma_c)        
    plt.figure()
    plt.plot(overall_hist)
    plt.xlabel("pre neurons")
    plt.ylabel("# of connections")
    plt.twinx()    
    plt.plot(true_function, color="red", linewidth=3)
    plt.title("Histograms of all pre neurons rotated to same configuration")
