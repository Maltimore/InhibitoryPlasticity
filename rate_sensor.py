import numpy as np
import mytools
import matplotlib.pyplot as plt




# This script assumes that the simulation has been run before with
# a simtime of 10000 ms.
t = 10000*ms

_, firing_rates = mytools.estimate_single_firing_rates(inhSpikeMon, 
                         rate_interval, simtime,
                         t_min = t - rate_interval, t_max = t)


def _exp_function(x_vec, mu, scale):
    # catch the case where scale == 0
    if scale == 0:
        y_vec = np.zeros(len(x_vec))
        y_vec[x_vec==mu] = 1
        return y_vec
    #else, compute normal exponential function
    return 1/scale * np.exp(2* -np.abs(x_vec - mu) / scale)

# keep in mind that mu has to be converted from neuron index to spatial
# first.
x_NI = 4
mu = 1000 * x_NI
sigma_s = 500
x_vec = np.arange(2000) * x_NI
y_vec = _exp_function(x_vec, mu, sigma_s)
y_vec /= np.sum(y_vec)

plt.figure()
plt.plot(x_vec, y_vec)
plt.vlines(np.array([500, 1000, 1500])*x_NI, 0, np.amax(y_vec),
           linestyles="dashed")
plt.xlabel("Space along ring")
plt.ylabel("window weight")
plt.title("Exp. rate window, dashed lines represent 1 sigma_s to left / right")

neuron_idx = 1200
y_vec = np.roll(y_vec, neuron_idx - int(mu/x_NI))

plt.figure()
plt.plot(x_vec, y_vec)
plt.xlabel("Space along ring")
plt.ylabel("window weight")
plt.title("Rotated by neuron_idx")

plt.figure()
plt.plot(x_vec, firing_rates)
plt.ylabel("Rate [Hz]")
plt.xlabel("Space along ring")
plt.twinx()
plt.plot(x_vec, y_vec, color="red", linewidth=3)
sensor_rate = np.dot(y_vec, firing_rates)
plt.title("The rate of this particular window is: " + str(sensor_rate))



def rate_sensor(firing_rates, x_NI, sigma_s):
    """ Compute the firing rates per neuron with an exponential window across
        the neighboring neurons. The exponential window is paraemtrized by
        sigma_s, which is the width of the exponential. """
    N_neurons = len(firing_rates)
    # Creating the exponential window and set its maximum over the "middle"
    # (in the vector) neuron. Later we will just take this window and rotate
    # it according to our needs (according to which neurons firing rate we're
    # estimating).
    mu = int(N_neurons/2) * x_NI
    x_vec = np.arange(N_neurons) * x_NI
    y_vec = _exp_function(x_vec, mu, sigma_s) * x_NI
    y_vec /= np.sum(y_vec)
    
    sensor_rates = np.zeros(N_neurons)
    for neuron_idx in np.arange(N_neurons):
        y_vec_temp = np.roll(y_vec, int(neuron_idx - (mu/x_NI)))
        
        sensor_rates[neuron_idx] = np.dot(y_vec_temp, firing_rates)
    
    return sensor_rates

sensor_rates = rate_sensor(firing_rates, x_NI, sigma_s)
plt.figure()
plt.plot(x_vec, sensor_rates)
plt.ylim([0, np.amax(sensor_rates)+1])
plt.xlabel("Space along ring")
plt.ylabel("Rate [Hz]")
plt.title("Sensor sensed rates with sigma_s = " + str(sigma_s))