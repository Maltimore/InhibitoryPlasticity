import sys
import numpy as np

n_sensors = 13
n_connectivities = 5
n_total = n_sensors * n_connectivities

sensor_widths = list(2 ** np.arange(12))
sensor_widths[0] = 0
sensor_widths.append("infinity")
sensor_widths = [val for val in sensor_widths for _ in range(n_connectivities)]

connectivities = [100, 300, 600, 1000, "infinity"] * n_sensors

all_args = list(zip(sensor_widths, connectivities))



if len(sys.argv) < 2:
    print("Call for simulation didn't receive enough arguments.")
    print("Exiting..")
elif len(sys.argv) > 2:
    print("Call for simulation received too many arguments.")
    print("Exiting..")
elif int(sys.argv[1]) > n_total:
    # notice that qsub indexes from 1 therefor it's not ">="
    print("Index " + str(sys.argv[1]) + " doesn't match any parameter set.")
    print("Exiting..")
else:
    index = int(sys.argv[1]) - 1 # qsub cannot give task ID 0.
    sigma_s = all_args[index][0]
    sigma_c = all_args[index][1]
    
    print("I'm running a simulation with sigma_s = " + str(sigma_s) + \
          " and sigma_c = " + str(sigma_c))
    