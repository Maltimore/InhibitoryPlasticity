from brian2 import *
import pickle

results = pickle.load( open( "results.p", "rb" ) )

#spike_times = results["spike_times"] * second
#spike_neurons = results["spike_neurons"]

inhWeights = results["inhWeights"]
weight_times = results["weight_times"]