import numpy as np
import matplotlib.pyplot as plt

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
    
    
fixed_in_degree = 0.02
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
    rand_pre_positions = np.random.exponential(scale=100, size=k_in)
    rand_pre_positions += 1 # so a neuron doesn't project onto itself
    rand_pre_positions = rand_pre_positions * \
                         (np.random.randint(0, 2, size=k_in)*2 - 1)
    rand_pre_positions += post_idx * x_post
    current_pre_neurons = np.zeros(k_in)
    for pos_idx, curr_pre_pos in enumerate(rand_pre_positions):
        while curr_pre_pos > N_pre:
            curr_pre_pos -= N_pre
        while curr_pre_pos < 0:
            curr_pre_pos += N_pre
        current_pre_neurons[pos_idx] = find_nearest(pre_positions,
                                                    curr_pre_pos)
    post_neurons[k_in*post_idx:k_in*(post_idx+1)] = \
        np.ones(k_in) * post_idx
    pre_neurons[k_in*post_idx:k_in*(post_idx+1)] = current_pre_neurons
post_neurons = post_neurons.astype(int)
pre_neurons = pre_neurons.astype(int)

plt.figure()
plt.hist(current_pre_neurons)


# checking whether the algorithm works
for pre_neuron_idx in np.arage(N_pre):
    pre_idxes = np.arange(pre_neuron_idx*k_in, (pre_neuron_idx+1)*k_in)
    curr_pre_neurons = pre_neurons[pre_idxes]
    curr_pre_neurons -= 1
    