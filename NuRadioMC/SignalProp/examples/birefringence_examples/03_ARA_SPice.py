from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units
import NuRadioReco.framework.electric_field
import matplotlib.pyplot as plt
import numpy as np
import time

from NuRadioMC.SignalProp import radioproparaytracing
from NuRadioMC.SignalProp import analyticraytracing

#Left to do

# correct emitter and reciver positions
# correct emitter angle
# fold antenna response
# calculate max of hilbert envelope



SPice_position = np.array([0 , 0 , -1300 ])* units.m
A5ara_position = np.array([-400, 4000, -200])* units.m

ref_index_model = 'southpole_2015'
ice = medium.get_ice_model(ref_index_model)
rays = analyticraytracing.ray_tracing(ice)
ray_tracing_solution = 0





re_sr = 2*10**(9) * units.hertz

config = {'propagation': {}}
config['propagation']['attenuate_ice'] = True
config['propagation']['focusing_limit'] = 2
config['propagation']['focusing'] = False
config['propagation']['birefringence'] = True


propagation_pulse = NuRadioReco.framework.electric_field.ElectricField([1], position=None,
                 shower_id=None, ray_tracing_id=None)


def birefringence_propagation(e_field, emitter_depth):

    propagation_pulse.set_trace(e_field, sr)
    propagation_pulse.resample(re_sr)
    SPice_position = np.array([0 , 0 , emitter_depth ])* units.m
    
    rays.set_start_and_end_point(SPice_position, A5ara_position)
    rays.find_solutions()
    rays.set_config(config)

    rays.apply_propagation_effects(propagation_pulse, ray_tracing_solution)

    vpol_max = np.max(propagation_pulse.get_trace()[1])
    hpol_max = np.max(propagation_pulse.get_trace()[2])

    return vpol_max, hpol_max




depths = np.arange(-1600, -849, 10)
pulse_number = np.arange(0, 10, 1)
print(depths)

v_max = np.empty(shape=(len(pulse_number), len(depths)))
h_max = np.empty(shape=(len(pulse_number), len(depths)))


for waveform in pulse_number:

    
    pulse = np.load('SPice_pulses/eField_launchAngle_15_set_' + str(waveform) + '.npy')
    dt = pulse[0, 1] - pulse[0, 0]
    sr = 1 / dt

    r_component = np.zeros(len(pulse[1]))
    electric_field = np.stack((r_component, pulse[1], pulse[2]))

    for depth in range(len(depths)):

        print('waveform: ', waveform)
        print('depth: ', depths[depth])

        st = time.time()
        vpol_max, hpol_max = birefringence_propagation(electric_field, depths[depth])
        et = time.time()
        print('Execution time (numerical propagation):', et - st, 'seconds')

        v_max[waveform, depth] = vpol_max
        h_max[waveform, depth] = hpol_max




plt.plot(depths, np.mean(v_max, axis=0), label='pulse theta')
plt.plot(depths, np.mean(h_max, axis=0), label='pulse phi')


plt.legend()
plt.show()