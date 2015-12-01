""" When saving values to disk, we are not taking into account Brian units.
    Threfore we are assigning standard units so that when recovering the 
    saved files, one knows which units to assign to them.
    time: second
    weights: nS
"""

from brian2 import *

start_scope()

N = 100
tau = 10*ms
eqs = '''
dv/dt = (2-v)/tau : 1
'''

G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0')
G.v = 'rand()'

spikemon = SpikeMonitor(G)

run(50*ms)

plot(spikemon.t/ms, spikemon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')