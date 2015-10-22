import numpy as np
import matplotlib.pyplot as plt

def find_nearest(array,value):
    distances = (np.abs(array-value))
    idx = np.argmin(distances)
    return idx
    

fixed_in_degree = 0.02
sigma_c = 50
N_pre = 2000
N_post = 8000
x_pre = 4
x_post = 1
pre_idxes = np.arange(N_pre)
post_idxes = np.arange(N_post)
pre_positions = pre_idxes * x_pre
post_positions = post_idxes * x_post

k_in = int(fixed_in_degree * N_pre)
pre_neurons = np.zeros(k_in * N_post)
post_neurons = np.zeros(k_in * N_post)
for post_idx in post_idxes:
    # draw from an exponential distribution
    rand_pre_positions = np.random.exponential(scale=sigma_c, size=k_in)
    rand_pre_positions += 1 # so a neuron doesn't project onto itself
    rand_pre_positions *= (np.random.randint(0, 2, size=k_in)*2 - 1)
    rand_pre_positions += post_idx * x_post
    current_pre_neurons = np.zeros(k_in)
    for pos_idx, curr_pre_pos in enumerate(rand_pre_positions):
        while curr_pre_pos > N_pre * x_pre:
            curr_pre_pos -= N_pre * x_pre
        while curr_pre_pos * x_pre < 0:
            curr_pre_pos += N_pre * x_pre
        current_pre_neurons[pos_idx] = find_nearest(pre_positions,
                                                    curr_pre_pos)
    post_neurons[k_in*post_idx:k_in*(post_idx+1)] = \
        np.ones(k_in) * post_idx
    pre_neurons[k_in*post_idx:k_in*(post_idx+1)] = current_pre_neurons
post_neurons = post_neurons.astype(int)
pre_neurons = pre_neurons.astype(int)


# checking whether the algorithm works
n_bins = N_pre
bin_edges = np.linspace(0, N_pre, n_bins+1)
post_neuron_idx = 0
hist_mat = np.zeros((N_post, n_bins))
overall_hist = np.zeros(n_bins)
for post_neuron_idx in np.arange(N_post):
    pre_idxes = pre_neurons[post_neuron_idx*k_in : (post_neuron_idx+1)*k_in]
    hist, bin_edges = np.histogram(pre_idxes, bins=bin_edges)
    hist_mat[post_neuron_idx, :] = hist

    hist = np.roll(hist, int(n_bins/2)-int(post_neuron_idx/4))
    overall_hist += hist

plt.figure()
plt.pcolormesh(hist_mat)

plt.figure()
plt.plot(overall_hist)