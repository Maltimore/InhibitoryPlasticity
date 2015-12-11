#from brian2 import *
import pickle
import mytools
import os
import numpy as np
import matplotlib.pyplot as plt


prep_time = 2000 # seconds


program_dir = os.getcwd()
lookuptable = np.array(mytools.lookuptable())
all_sigma_s = np.sort(np.array(list(set(lookuptable[:,0])))) / 2
all_sigma_c = np.sort(np.array(list(set(lookuptable[:,1])))) / 2
n_sigma_s = len(all_sigma_s)
n_sigma_c = len(all_sigma_c)



# loop over parameter space
sparseness_vec = np.empty(len(lookuptable))
sparseness_vec[:] = np.NaN
sq_error_vec = np.empty(len(lookuptable))
sq_error_vec[:] = np.NaN
avg_rate_vec = np.empty(len(lookuptable))
avg_rate_vec[:] = np.NaN
for table_idx in np.arange(len(lookuptable)):
    sigma_s, sigma_c = lookuptable[table_idx,:]
    sigma_s /= 2
    sigma_c /= 2
    
    resultfile = "sigma_s_" + str(sigma_s) + "_" + \
                 "sigma_c_" + str(sigma_c) + "_" + \
                 "prep_" + str(int(prep_time)) + "_seconds"
    # open file
    try:
        results = pickle.load(open(program_dir + "/results/rates_and_weights/" 
                               + resultfile, "rb"))
        simtime = 100 # seconds (has to be seconds!!)
        rho_0 = 15 # Hz
        simtime = results["simtime"]
        rho_0 = results["rho_0"]
    except:
        print("Failed for table index " + str(table_idx) +
              " with sigma_s = " + str(sigma_s) + 
              ", sigma_c = " + str(sigma_c))
        continue
    
    # loop over timesteps
    tmp_sparseness = 0
    tmp_sq_error = 0
    for timestep in np.arange(int(simtime)):
        rates = results["inh_rates"][:,timestep]
        tmp_sparseness += mytools.compute_sparseness(rates)
        tmp_sq_error += np.average(np.square(rates - rho_0))
    sparseness_vec[table_idx] = tmp_sparseness / int(simtime)
    sq_error_vec[table_idx] = tmp_sq_error / int(simtime)
    avg_rate_vec[table_idx] = np.average(results["inh_rates"])
    
sparseness_vec_m = np.ma.array (sparseness_vec, mask=np.isnan(sparseness_vec))
sq_error_vec_m = np.ma.array (sq_error_vec, mask=np.isnan(sq_error_vec))
avg_rate_vec_m = np.ma.array (avg_rate_vec, mask=np.isnan(avg_rate_vec))

# it is important to remember that the lookuptable first loops over the
# sigma_c
sparseness_mat = np.reshape(sparseness_vec_m, (n_sigma_s, n_sigma_c))
sq_error_mat = np.reshape(sq_error_vec_m, (n_sigma_s, n_sigma_c))
avg_rate_mat = np.reshape(avg_rate_vec_m, (n_sigma_s, n_sigma_c))

def plot_heatmap(data, all_sigma_s, all_sigma_c, invert=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    if invert:
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues_r)
    else:
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_xticklabels(all_sigma_c, minor=False)
    ax.set_yticklabels(all_sigma_s, minor=False)
    ax.set_xlabel("sigma c")
    ax.set_ylabel("sigma s")
    fig.colorbar(heatmap)
    return ax

ax = plot_heatmap(sparseness_mat, all_sigma_s, all_sigma_c, invert=True)
ax.set_title("Sparseness")
ax = plot_heatmap(sq_error_mat, all_sigma_s, all_sigma_c)
ax.set_title("Squared error")
ax = plot_heatmap(avg_rate_mat, all_sigma_s, all_sigma_c)
ax.set_title("Average rates")


rate_per_diffusion = np.ma.average(avg_rate_mat, axis=1)
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(rate_per_diffusion)
ax.set_xticks(np.arange(rate_per_diffusion.shape[0]))
ax.set_xticklabels(all_sigma_s)
ax.set_xlabel("Diffusion width")
ax.set_ylabel("Rates [Hz]")
ax.set_title("Rate per diffusion")


#matrix_axis = np.floor(np.sqrt(len(rate_vector)))
#rate_vector = rate_vector[:matrix_axis**2]
#rate_mat = np.reshape(rate_vector, (int(np.sqrt(N_inh_neurons)), -1))
#fig, ax = plt.subplots()
#ax.pcolor(rate_mat, cmap="Reds")
#plt.title("Inh firing rate estimated with counting spikes")
#plt.xticks([]); plt.yticks([]);
#
#plt.show()