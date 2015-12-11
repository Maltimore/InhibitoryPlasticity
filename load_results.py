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

sparseness_vec = np.ones(len(lookuptable))
for table_idx in np.arange(len(lookuptable)):
    sigma_s, sigma_c = lookuptable[table_idx,:]
    sigma_s /= 2
    sigma_c /= 2
    
    resultfile = "sigma_s_" + str(sigma_s) + "_" + \
                 "sigma_c_" + str(sigma_c) + "_" + \
                 "prep_" + str(int(prep_time)) + "_seconds"
    
    try:
        results = pickle.load(open(program_dir + "/results/rates_and_weights/" 
                               + resultfile, "rb"))
    except:
        print("Failed for table index " + str(table_idx) +
              " with sigma_s = " + str(sigma_s) + 
              ", sigma_c = " + str(sigma_c))
        sparseness = np.NaN
        sparseness_vec[table_idx] = sparseness
        continue
    
    rates = results["inh_rates"][:,-1]
    sparseness = mytools.compute_sparseness(rates)
    sparseness_vec[table_idx] = sparseness
sparseness_vec_m = np.ma.array (sparseness_vec, mask=np.isnan(sparseness_vec))

# it is important to remember that the lookuptable first loops over the
# sigma_c
sparseness_mat = np.reshape(sparseness_vec_m, (n_sigma_s, n_sigma_c))

fig, ax = plt.subplots(figsize=(8, 8))
heatmap = ax.pcolor(sparseness_mat, cmap=plt.cm.Blues)
# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(sparseness_mat.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(sparseness_mat.shape[1])+0.5, minor=False)
ax.set_xticklabels(all_sigma_c, minor=False)
ax.set_yticklabels(all_sigma_s, minor=False)
ax.set_xlabel("sigma c")
ax.set_ylabel("sigma s")
cb = fig.colorbar(heatmap)