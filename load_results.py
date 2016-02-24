from brian2 import second, ms
import pickle
import mytools
import os
import numpy as np
import matplotlib.pyplot as plt

dataset = "simtime_20000_nonreversed_rho0_15Hz"
#dataset = "fullresult_nonreversed_normal_rho0_7Hz"
verbose = False
fullresult_mode = False
do_histograms = False
my_fontsize=20
use_dpi = 400

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

def plot_heatmap(data, all_sigma_s, all_sigma_c, invert=False, title="",
                 fontsize=16):
    fig, ax = plt.subplots(figsize=(8, 8))
    if invert:
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues_r)
    else:
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
    xticks = list(all_sigma_c.astype(int)[:-1])
    xticks.append("inf")
    yticks = list(all_sigma_s.astype(int)[:-1])
    yticks.append("inf")

    ax.set_xticklabels(xticks, minor=False)
    ax.set_yticklabels(yticks, minor=False)
    ax.set_xlabel("connectivity spread $\sigma_c$", fontsize=fontsize)
    ax.set_ylabel("sensor width $\sigma_s$", fontsize=fontsize)
    ax.tick_params(labelsize=16)
#    if title != "":
#        ax.set_title(title)
    cb = fig.colorbar(heatmap)
    cb.ax.tick_params(labelsize=16)
    plt.savefig(plots_dir + title + "_rho0_" + str(rho_0) + "Hz.png", dpi=use_dpi)
    return ax

if not fullresult_mode:
    # Sparseness plot
    ax = plot_heatmap(sparseness_mat, all_sigma_s, all_sigma_c, invert=True,
                      title="Sparseness")
    # Squared error plot
    ax = plot_heatmap(sq_error_mat, all_sigma_s, all_sigma_c,
                      title="Squared_error", fontsize=my_fontsize)
    # Average rates plot
    ax = plot_heatmap(avg_rate_mat, all_sigma_s, all_sigma_c,
                      title="Average_rates", fontsize=my_fontsize)
    # Min weights plot
    ax = plot_heatmap(n_min_weights, all_sigma_s, all_sigma_c,
                      title="Number_of_minimum_weights", fontsize=my_fontsize)
    # Max weights plot
    ax = plot_heatmap(n_max_weights, all_sigma_s, all_sigma_c,
                      title="Number_of_max_weights", fontsize=my_fontsize)


    # Firing rate per diffusion
    rate_per_diffusion = np.ma.average(avg_rate_mat, axis=1)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(rate_per_diffusion, 'bo', rate_per_diffusion, 'k')
    ax.set_xticks(np.arange(rate_per_diffusion.shape[0]))
    ax.set_xticklabels(all_sigma_s, fontsize=my_fontsize)
    ax.tick_params(labelsize=my_fontsize)
    ax.set_xlabel("Diffusion width", fontsize=my_fontsize)
    ax.set_ylabel("Rates [Hz]", fontsize=my_fontsize)
#    ax.set_title("Rate per diffusion")
    ax.set_ylim([np.amin(rate_per_diffusion)-1, np.amax(rate_per_diffusion)+1])
    plt.savefig(plots_dir + "Rate_per_diffusion_rho0_" + str(rho_0) + "Hz.png", dpi=600)
    
if not fullresult_mode and do_histograms:    
    # Weight histograms
    fig, axes = plt.subplots(n_sigma_c, n_sigma_s, figsize=(15, 15),
                             sharex=True, sharey=True)
    for sigma_c_idx, row in enumerate(axes.T):
        for sigma_s_idx, ax in enumerate(row[::-1]):
            hist = weight_hist[sigma_s_idx, sigma_c_idx]
            ax.bar(weight_bin_edges[:-1], hist, width = bin_width - .01)
            ax.set_xticks([])
    #        ax.set_yticks([])
            ax.set_ylim([0, 10000])
            if sigma_c_idx == 0:
                ax.set_ylabel(all_sigma_s[sigma_s_idx], fontsize=18)
            if sigma_s_idx == 0:
                ax.set_xlabel(all_sigma_c[sigma_c_idx], fontsize=18)
    plt.tight_layout()
    plt.savefig(plots_dir + "Weight histograms_rho0_" + str(rho_0) + "Hz.png", dpi=use_dpi)
    
    
    
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
    plt.savefig(plots_dir + "rate histograms_rho0_" + str(rho_0) + "Hz.png", dpi=use_dpi)
#    plt.suptitle("Inhibitory rate histograms")



# RASTER PLOTS
if fullresult_mode:
    for table_idx in np.arange(len(lookuptable)):
        sigma_s, sigma_c = lookuptable[table_idx,:]
    
        resultfile = "sigma_s_" + str(sigma_s) + "_" + \
                     "sigma_c_" + str(sigma_c) + "_" + \
                     "prep_" + str(int(prep_time)) + "_seconds"
        # open file
        try:
            results = pickle.load(open(results_dir + "/" + resultfile, "rb"))
            results["sigma_s"] = sigma_s
            results["sigma_c"] = sigma_c
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
 
    # RATE OVER TIME
    # loop over timesteps
    plot_n_timesteps = 10
    avg_rates = np.empty(plot_n_timesteps)
    for timestep in np.arange(1, plot_n_timesteps+1):
        rates = results["inh_rates"][:,timestep]
        avg_rates[timestep-1] = np.average(rates)
    plt.figure()
    plt.plot(np.arange(1, plot_n_timesteps+1), avg_rates)
    plt.xlabel("time [s]")
    plt.ylabel("rate [Hz]")
    
    spikes, bins = np.histogram(inh_spike_times[inh_spike_times<prep_time/secod + 5], bins=500)
    spikes = spikes.astype(float)/10

    rho_0 = results["rho_0"]    
    inh_spike_idxes = results["inh_spike_neuron_idxes"]
    inh_spike_times = results["inh_spike_times"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))
    ax1.plot(bins[:-1], spikes, linewidth=.5)
    ax1.set_ylim([0, 10])
    ax1.tick_params(labelsize=16)
    ax1.set_ylabel("Firing rate [Hz]", fontsize=my_fontsize)
    ax2.plot(inh_spike_times, inh_spike_idxes, '.k', markersize=2)
    ax2.set_xlabel('Time [s]', fontsize=my_fontsize)
    ax2.set_ylabel('Neuron index', fontsize=my_fontsize)
    ax2.set_xlim([prep_time/second, (prep_time)/second +5])
    ax2.set_ylim([0,100])
    ax2.tick_params(labelsize=16)
#    ax2.title("Raster plot of firing in inh cells")
    plt.savefig(plots_dir + "inh_raster_plot_rho0_" + str(rho_0) + "Hz.png",
                dpi=use_dpi)
    
    exc_spike_idxes = results["exc_spike_neuron_idxes"]
    exc_spike_times = results["exc_spike_times"]
    
    plt.figure()
    plt.plot(exc_spike_times, exc_spike_idxes, '.k')
    plt.xlabel('Time [s]')
    plt.ylabel('Neuron index')
    plt.xlim([prep_time/second, (prep_time)/second +3])
    plt.ylim([0,100])
#    plt.title("Raster plot of firing in exc cells")
    plt.savefig(plots_dir + "exc_raster_plot_rho0_" + str(rho_0) + "Hz.png", dpi=use_dpi)
    
    
    
   

# fullresult mode
if fullresult_mode:
    conn_filename = mytools._create_connectivity_filename("inh_to_exc",
                                                      results["sigma_c"],
                                                      1000,
                                                      4000)

    i_to_e = pickle.load(open(program_dir + "/connectivity_matrices/" +
                                conn_filename, "rb"))
    conn_filename = mytools._create_connectivity_filename("exc_to_inh",
                                                      results["sigma_c"],
                                                      4000,
                                                      1000)
    e_to_i = pickle.load(open(program_dir + "/connectivity_matrices/" +
                                conn_filename, "rb"))
    # CALCULATING FEEDBACK CONNECTIONS
    inh_feedbacks = np.empty(results["NI"])
    for idx_neuron in np.arange(results["NI"]):
        # loop over all inhibitory neurons
        feedback = 0
        idx_neurons_projections = i_to_e[i_to_e[:,0]==idx_neuron, 1]
        
        for post_neuron in idx_neurons_projections:
            # loop over all the connections that the current index neuron has
            
            post_neurons_projections = e_to_i[e_to_i[:,0]==post_neuron, 1]
            if idx_neuron in post_neurons_projections:
                # if the current postsynaptic neuron has a connection back to
                # the original neuron, increment feedback
                feedback += 1
        inh_feedbacks[idx_neuron] = feedback
        

    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(inh_feedbacks, rates)
    print("The p value for the regression test for feedback connections" +
          " and rates is " + str(p_value))
    plt.figure()
    plt.scatter(inh_feedbacks, rates)
    plt.xlabel("Number of feedback connections")
    plt.ylabel("Rate of neuron")

    
    errors = np.square(rates-results["rho_0"])
    slope, intercept, r_value, p_value, std_err = stats.linregress(inh_feedbacks, errors)
    print("The p value for the regression test for feedback connections" +
          " and square errors is " + str(p_value))
    plt.figure()
    plt.scatter(inh_feedbacks, rates)
    plt.xlabel("Number of feedback connections")
    plt.ylabel("Squared error")

    
    
    at_least_n_datapoints = 20
    
    avg_rate = []
    SE_rate = []
    avg_error = []
    SE_error = []
    for idx in range(20):
        n_values = np.sum(inh_feedbacks==idx)
        if n_values < at_least_n_datapoints:
            show_n_points = idx
            break
        avg_rate.append(np.average(rates[inh_feedbacks==idx]))
        SE_rate.append(np.std(rates[inh_feedbacks==idx]) / np.sqrt(n_values))
        avg_error.append(np.average(errors[inh_feedbacks==idx]))
        SE_error.append(np.std(errors[inh_feedbacks==idx]) / np.sqrt(n_values))

    plt.figure()
    plt.errorbar(range(show_n_points), avg_rate, yerr=SE_rate)
    plt.xlim([-1,show_n_points])
    plt.xlabel("N feedback connections")
    plt.ylabel("Avg rate")
    
    plt.figure()
    plt.errorbar(range(show_n_points), avg_error, yerr=SE_error)
    plt.xlim([-1,show_n_points])
    plt.xlabel("N feedback connections")
    plt.ylabel("Avg squared error")
    
    n_bins = 15
    rate_hist_mat = np.empty((n_bins, show_n_points))
    # create one histogram per "column" (per # of feedback connections)
    for idx in np.arange(show_n_points):
        rate_hist, rate_bins = np.histogram(rates[inh_feedbacks==idx],
                                            bins=np.linspace(0, np.amax(rates), n_bins+1))
        # normalize:
        rate_hist = rate_hist.astype(float) / np.amax(rate_hist)
        rate_hist_mat[:, idx] = rate_hist
    rate_hist_mat = rate_hist_mat
    
    
    fig, ax = plt.subplots(figsize=(8, 8))
    heatmap = ax.pcolor(rate_hist_mat, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(rate_hist_mat.shape[1])+0.5, minor=False)
    ax.set_xticklabels(np.arange(show_n_points), minor=False)
    ax.set_yticks(range(len(rate_bins)))
    ax.set_yticklabels(rate_bins, minor=False)
    ax.set_ylim([0, len(rate_bins)-1])
    ax.set_xlabel("# feedback connections")
    ax.set_ylabel("rate [Hz]")
    fig.colorbar(heatmap)
    plt.savefig(plots_dir + "feedback_rate_hist" + ".png", dpi=use_dpi)
    
    # RATE HISTOGRAM    
    fix, ax = plt.subplots(figsize=(8, 8))
    ax.hist(rates, bins=15)
    ax.set_xlabel("Firing rate [Hz]", fontsize=my_fontsize)
    ax.set_ylabel("# of inhibitory cells", fontsize=my_fontsize)
    ax.tick_params(labelsize=my_fontsize)
#    ax.set_title("Rate histogram of inhibitory cells")
    plt.savefig(plots_dir + "inh_rate_histogram_rho0_" + str(rho_0) + "Hz.png", dpi=use_dpi)
    
