from brian2 import second
import pickle
import mytools
import os
import numpy as np
import matplotlib.pyplot as plt


dataset = "exc_50_bg_rho0_7Hz"
verbose = False
fullresult_mode = False

program_dir = os.getcwd()
results_dir = program_dir + "/results/" + dataset
plots_dir = program_dir + "/plots/" + dataset + "/"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load the parameters file
try:
    print("Trying to load parameter dataset from path")
    print(results_dir)
    params_file = pickle.load(open(results_dir + "/parameter_file", "rb"))
    print("Success!")
except:
    print("Failed to load the parameter dataset!")


lookuptable = np.array(params_file["lookuptable"])
simtime = params_file["simtime"] / second
prep_time = params_file["prep_time"] / second
rho_0 = params_file["rho_0"]
w_min = params_file["wmin"]
w_max = params_file["wmax"]

all_sigma_s = np.sort(np.array(list(set(lookuptable[:,0]))))
all_sigma_c = np.sort(np.array(list(set(lookuptable[:,1]))))
n_sigma_s = len(all_sigma_s)
n_sigma_c = len(all_sigma_c)
n_weight_bins = 30
n_rate_bins = 30
bin_width = (w_max - w_min) / n_weight_bins
max_rate_bin = 50
rate_bin_width = max_rate_bin / n_rate_bins
use_dpi = 400

# loop over parameter space
sparseness_vec = np.empty(len(lookuptable))
sparseness_vec[:] = np.NaN
sq_error_vec = np.empty(len(lookuptable))
sq_error_vec[:] = np.NaN
avg_rate_vec = np.empty(len(lookuptable))
avg_rate_vec[:] = np.NaN
n_min_weights = np.empty(len(lookuptable))
n_min_weights[:] = np.NaN
n_max_weights = np.empty(len(lookuptable))
n_max_weights[:] = np.NaN
weight_hist = np.empty((n_sigma_s, n_sigma_c, n_weight_bins))
weight_hist[:] = np.NaN
rate_hist = np.empty((n_sigma_s, n_sigma_c, n_rate_bins))
rate_hist[:] = np.NaN
for table_idx in np.arange(len(lookuptable)):
    sigma_s, sigma_c = lookuptable[table_idx,:]

    resultfile = "sigma_s_" + str(sigma_s) + "_" + \
                 "sigma_c_" + str(sigma_c) + "_" + \
                 "prep_" + str(int(prep_time)) + "_seconds"
    # open file
    try:
        results = pickle.load(open(results_dir + "/" + resultfile, "rb"))
        simtime = results["simtime"]
        rho_0 = results["rho_0"]
    except:
        if verbose:
            print("Failed for table index " + str(table_idx) +
                  " with sigma_s = " + str(sigma_s) +
                  ", sigma_c = " + str(sigma_c))
            print("To restart the simulation, remember that the qsub index is " +
                  str(table_idx + 1))
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
    n_min_weights[table_idx] = len(np.where(results["inhWeights"] < w_min + .01)[0])
    n_max_weights[table_idx] = len(np.where(results["inhWeights"] > w_max - .01)[0])

    w_hist, weight_bin_edges = np.histogram(results["inhWeights"], n_weight_bins,
                                   range=(w_min, w_max))
    weight_hist[all_sigma_s == sigma_s, all_sigma_c == sigma_c, :] = w_hist

    r_hist, rate_bin_edges = np.histogram(results["inh_rates"][:,-1], n_rate_bins,
                                             range=(0, max_rate_bin))
    rate_hist[all_sigma_s == sigma_s, all_sigma_c == sigma_c, :] = r_hist



# masking arrays for NaN values
sparseness_vec_m = np.ma.array(sparseness_vec, mask=np.isnan(sparseness_vec))
sq_error_vec_m = np.ma.array(sq_error_vec, mask=np.isnan(sq_error_vec))
avg_rate_vec_m = np.ma.array(avg_rate_vec, mask=np.isnan(avg_rate_vec))
n_min_weights = np.ma.array(n_min_weights, mask=np.isnan(n_min_weights))
n_max_weights = np.ma.array(n_max_weights, mask=np.isnan(n_max_weights))


# it is important to remember that the lookuptable first loops over the
# sigma_c
sparseness_mat = np.reshape(sparseness_vec_m, (n_sigma_s, n_sigma_c))
sq_error_mat = np.reshape(sq_error_vec_m, (n_sigma_s, n_sigma_c))
avg_rate_mat = np.reshape(avg_rate_vec_m, (n_sigma_s, n_sigma_c))
n_min_weights = np.reshape(n_min_weights, (n_sigma_s, n_sigma_c))
n_max_weights = np.reshape(n_max_weights, (n_sigma_s, n_sigma_c))

def plot_heatmap(data, all_sigma_s, all_sigma_c, invert=False, title=""):
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
    if title != "":
        ax.set_title(title)
    fig.colorbar(heatmap)
    plt.savefig(plots_dir + title + ".png", dpi=use_dpi)
    return ax

if not fullresult_mode:
    # Sparseness plot
    ax = plot_heatmap(sparseness_mat, all_sigma_s, all_sigma_c, invert=True,
                      title="Sparseness")
    # Squared error plot
    ax = plot_heatmap(sq_error_mat, all_sigma_s, all_sigma_c, title="Squared error")
    # Average rates plot
    ax = plot_heatmap(avg_rate_mat, all_sigma_s, all_sigma_c, title="Average rates")
    # Min weights plot
    ax = plot_heatmap(n_min_weights, all_sigma_s, all_sigma_c,
                      title="Number of minimum weights")
    # Max weights plot
    ax = plot_heatmap(n_max_weights, all_sigma_s, all_sigma_c,
                      title="Number of max weights")


# Firing rate per diffusion
#rate_per_diffusion = np.ma.average(avg_rate_mat, axis=1)
#fig, ax = plt.subplots(figsize=(8, 8))
#ax.plot(rate_per_diffusion, 'bo', rate_per_diffusion, 'k')
#ax.set_xticks(np.arange(rate_per_diffusion.shape[0]))
#ax.set_xticklabels(all_sigma_s)
#ax.set_xlabel("Diffusion width")
#ax.set_ylabel("Rates [Hz]")
#ax.set_title("Rate per diffusion")
#ax.set_ylim([np.amin(rate_per_diffusion)-1, np.amax(rate_per_diffusion)+1])
#plt.savefig(plots_dir + "Rate per diffusion" + ".png", dpi=use_dpi)
#
# Weight histograms
#fig, axes = plt.subplots(n_sigma_c, n_sigma_s, figsize=(15, 15),
#                         sharex=True, sharey=True)
#for sigma_c_idx, row in enumerate(axes.T):
#    for sigma_s_idx, ax in enumerate(row[::-1]):
#        hist = weight_hist[sigma_s_idx, sigma_c_idx]
#        ax.bar(weight_bin_edges[:-1], hist, width = bin_width - .01)
#        ax.set_xticks([])
##        ax.set_yticks([])
#        ax.set_ylim([0, 10000])
#        if sigma_c_idx == 0:
#            ax.set_ylabel(all_sigma_s[sigma_s_idx], fontsize=18)
#        if sigma_s_idx == 0:
#            ax.set_xlabel(all_sigma_c[sigma_c_idx], fontsize=18)
#plt.tight_layout()
#plt.savefig(plots_dir + "Weight histograms.png", dpi=use_dpi)
#
#
#
# Rate histograms
fig, axes = plt.subplots(n_sigma_c, n_sigma_s, figsize=(15, 15),
                         sharex=True, sharey=True)
for sigma_c_idx, row in enumerate(axes.T):
    for sigma_s_idx, ax in enumerate(row[::-1]):
        hist = rate_hist[sigma_s_idx, sigma_c_idx]
        ax.bar(rate_bin_edges[:-1], hist, width = rate_bin_width - .01)
        if sigma_c_idx == 0:
            ax.set_ylabel(all_sigma_s[sigma_s_idx], fontsize=18)
        if sigma_s_idx == 0:
            ax.set_xlabel(all_sigma_c[sigma_c_idx], fontsize=18)
plt.tight_layout()
plt.savefig(plots_dir + "rate histograms.png", dpi=use_dpi)
plt.suptitle("Inhibitory rate histograms")


if fullresult_mode:
    for table_idx in np.arange(len(lookuptable)):
        sigma_s, sigma_c = lookuptable[table_idx,:]
    
        resultfile = "sigma_s_" + str(sigma_s) + "_" + \
                     "sigma_c_" + str(sigma_c) + "_" + \
                     "prep_" + str(int(prep_time)) + "_seconds"
        # open file
        try:
            results = pickle.load(open(results_dir + "/" + resultfile, "rb"))
            print("Loaded full result dataset.")            
        except:
            if verbose:
                print("Failed for table index " + str(table_idx) +
                      " with sigma_s = " + str(sigma_s) +
                      ", sigma_c = " + str(sigma_c))
                print("To restart the simulation, remember that the qsub index is " +
                      str(table_idx + 1))
            continue
    prep_time = results["prep_time"]
    simtime = results["simtime"]
    
    rho_0 = results["rho_0"]    
    inh_spike_idxes = results["inh_spike_neuron_idxes"]
    inh_spike_times = results["inh_spike_times"]
    
    plt.plot(inh_spike_times, inh_spike_idxes, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.xlim([prep_time/second, (prep_time)/second +1])