from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

def create_neural_components():
    # Network model parameters
    NE = 400            # Number of excitatory cells
    NI = NE/4           # Number of inhibitory cells
    tau_ampa = 5.0*ms   # Glutamatergic synaptic time constant
    tau_gaba = 10.0*ms  # GABAergic synaptic time constant
    epsilon = 0.02      # Sparseness of synaptic connections
    tau_stdp = 20*ms    # STDP time constant
    simtime = 300*ms    # Simulation time
    
    # Neuron model
    gl = 10.0*nsiemens   # Leak conductance
    el = -60*mV          # Resting potential
    er = -80*mV          # Inhibitory reversal potential
    vt = -50.*mV         # Spiking threshold
    memc = 200.0*pfarad  # Membrane capacitance
    bgcurrent = 200*pA   # External current
    gmax = 100           # Maximum inhibitory weight
    eta = 0

    print("Creating Network components..")
    eqs_neurons='''
        dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er))+bgcurrent)/memc : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gaba/dt = -g_gaba/tau_gaba : siemens
        '''
    neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                          reset='v=el', refractory=5*ms)
    Pe = neurons[:NE]
    Pi = neurons[NE:]
    con_e = Synapses(Pe, neurons, pre='g_ampa += 0.3*nS', connect='rand()<epsilon')
    con_ii = Synapses(Pi, Pi, pre='g_gaba += 3*nS', connect='rand()<epsilon')
    neurons.v = el + 10*mV
    
    print("Adding inhibitory to excitatory connections..")
    eqs_stdp_inhib = '''
        w : 1
        pre_spikes_last_second : 1 
        '''
    con_ei = Synapses(network_dict["Pi"],
                      network_dict["Pe"],
                      model=eqs_stdp_inhib,
                      pre='''pre_spikes_last_second += 1.
                             g_gaba += w*nS
                             w += 1e-11''',
                      connect='rand()<epsilon')
    con_ei.w = 300
    con_ei.run_regularly("""pre_spikes_last_second = 0""", dt=1000*ms)

    print("Creating Network..")
    MyNet = Network(neurons, Pe, Pi, con_e, con_ii, con_ei, StateMon, VStateMon, SpikeMon, MyNet)

    return neurons, Pe, Pi, con_e, con_ii, con_ei, StateMon, VStateMon, SpikeMon, MyNet
  
neurons, Pe, Pi, con_e, con_ii, con_ei, StateMon, VStateMon, SpikeMon, MyNet = create_neural_components()
    

print("Adding inhibitory to excitatory connections..")
eqs_stdp_inhib = '''
    w : 1
    pre_spikes_last_second : 1 
    '''
con_ei = Synapses(network_dict["Pi"],
                  network_dict["Pe"],
                  model=eqs_stdp_inhib,
                  pre='''pre_spikes_last_second += 1.
                         g_gaba += w*nS
                         w += 1e-11''',
                  connect='rand()<epsilon')
con_ei.w = 300
con_ei.run_regularly("""pre_spikes_last_second = 0""", dt=1000*ms)
MyNet.add(con_ei)


StateMon = StateMonitor(con_ei, ['w', 'pre_spikes_last_second'], record=0)
SpikeMon = SpikeMonitor(network_dict["neurons"])
VStateMon = StateMonitor(network_dict["Pi"], 'v', record=0)
MyNet.store("initial_state")
MyNet.add(StateMon, SpikeMon, VStateMon)


print("Running simulation..")
MyNet.run(simtime, report="stdout")
print("Done simulating.")





#### PLOTTING

# spikes
plt.figure()
plt.plot(SpikeMon.t/ms, SpikeMon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')

# synaptic weight
plt.figure()
plt.plot(StateMon.t/ms, StateMon.w.T)
plt.xlabel("time [ms]")
plt.ylabel("weight")
plt.title("Synaptic weight of inhibitory synapse #1")

plt.figure()
plt.plot(StateMon.t/ms, StateMon.pre_spikes_last_second.T)
plt.xlabel("time [ms]")
plt.ylabel("# Spikes")
plt.title("Spikes this bin in the presynaptic neuron of inhibitory synapse #1")
plt.xlim([-10, np.amax(StateMon.t/ms)])
plt.ylim([0, np.amax(StateMon.pre_spikes_last_second)+1])

plt.figure()
plt.plot(VStateMon.t/ms, VStateMon.v.T)
plt.xlabel("time [ms]")
plt.ylabel("Voltage")
plt.title("Voltage in inhibitory neuron #1")