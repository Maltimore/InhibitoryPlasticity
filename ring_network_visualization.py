import numpy as np
import matplotlib.pyplot as plt
import os

def _find_nearest(array,value):
    distances = (np.abs(array-value))
    idx = np.argmin(distances)
    return idx
    

fixed_in_degree = 0.02
sigma_c = 500
N_pre = 8000
N_post = 2000
x_pre = 1
x_post = 4
pre_idxes = np.arange(N_pre)
post_idxes = np.arange(N_post)
post_positions = post_idxes * x_post

k_in = int(fixed_in_degree * N_pre)
pre_neurons = np.zeros(k_in * N_pre)
post_neurons = np.zeros(k_in * N_pre)

for pre_idx in pre_idxes:
    # draw from an exponential distribution (add 1 so no self projections)
    rand_post_positions = np.random.exponential(scale=sigma_c, size=k_in) + 1
    rand_post_positions *= (np.random.randint(0, 2, size=k_in)*2 - 1)
    rand_post_positions += pre_idx * x_pre
    post_neurons_helper = np.zeros(k_in)
    for pos_idx, curr_post_pos in enumerate(rand_post_positions):
        while curr_post_pos > N_post * x_post:
            curr_post_pos -= N_post * x_post
        while curr_post_pos * x_post < 0:
            curr_post_pos += N_post * x_post
        post_neurons_helper[pos_idx] = _find_nearest(post_positions,
                                                    curr_post_pos)
    pre_neurons[k_in*pre_idx:k_in*(pre_idx+1)] = \
        np.ones(k_in) * pre_idx
    post_neurons[k_in*pre_idx:k_in*(pre_idx+1)] = post_neurons_helper
connectivity_mat = np.vstack((pre_neurons, post_neurons)).astype(int).T






# checking whether the algorithm works by visualizing the connectivity
n_bins = 100
bin_edges = np.linspace(0, N_post, n_bins+1)
pre_neuron_idx = 0
hist_mat = np.zeros((N_pre, n_bins))
for pre_neuron_idx in pre_idxes:
    post_idxes = post_neurons[pre_neuron_idx*k_in : (pre_neuron_idx+1)*k_in]
    hist, _ = np.histogram(post_idxes, bins=bin_edges)
    hist_mat[pre_neuron_idx, :] = hist

plt.figure()
plt.pcolormesh(hist_mat)
plt.xlabel("post bins")
plt.ylabel("pre neurons index")
plt.title("Connectivity hist colormap")


n_bins = N_post
bin_edges = np.linspace(0, N_post, n_bins+1)
pre_neuron_idx = 0
overall_hist = np.zeros(n_bins)
for pre_neuron_idx in pre_idxes:
    post_idxes = post_neurons[pre_neuron_idx*k_in : (pre_neuron_idx+1)*k_in]
    hist, _ = np.histogram(post_idxes, bins=bin_edges)
    hist = np.roll(hist, int(n_bins/2)-int(pre_neuron_idx/4))
    overall_hist += hist
    
    
plt.figure()
plt.plot(overall_hist)
plt.xlabel("post neurons")
plt.ylabel("# of connections")
plt.title("Histograms of all pre neurons rotated to same configuration")