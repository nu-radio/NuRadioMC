import numpy as np
import NuRadioReco.modules.io.eventReader as eventReader
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities import units
import matplotlib.pyplot as plt

"""
This script is in example how to read a .nur file. It will plot the electric field from the air shower and the voltage traces per antenna, if the station has
triggered. The example file 'cr_single_station.nur' is created by running the sim_cr_single_station.py script.

The structure of the .nur file is described in the documentation of NuRadioReco.
"""

label = [r'$E_r$', r'$E_\theta$',r'$E_\phi$']
linestyle = ['solid', 'solid', 'dashed']

nur_file_list = ['cr_single_station.nur']
evtReader = eventReader.eventReader()
evtReader.begin(filename=nur_file_list, read_detector=True)
det = evtReader.get_detector()
for i, evt in enumerate(evtReader.run()):
    event_id = evt.get_id()
    for sim_shower in evt.get_sim_showers():
        corePositionX, corePositionY = sim_shower.get_parameter(shp.core)[0], sim_shower.get_parameter(shp.core)[1]
        energy = (sim_shower.get_parameter(shp.energy))
        zenith = (sim_shower.get_parameter(shp.zenith))
        azimuth = (sim_shower.get_parameter(shp.azimuth))
        print(f'Shower has energy {energy:.2e}, zenith {zenith/units.deg:.0f}, azimuth {azimuth/units.deg:.0f} and core position ({corePositionX:.0f}, {corePositionY:.0f})')
        for sta in evt.get_stations():
            station_id = sta.get_id()
            det_station_position = det.get_absolute_position(station_id)
            distance_core = np.sqrt((corePositionX - det_station_position[0])**2 + (corePositionY - det_station_position[1])**2)
            print(f'Station {station_id} has position {det_station_position} and distance to core {distance_core:.0f}m')
            trigger = sta.get_triggers()
            for trigger_name in trigger.keys():
                sim_station = sta.get_sim_station()
                triggered = sta.has_triggered(trigger_name)
                if triggered:
                    print(f'Station {station_id} has triggered True with trigger {trigger_name}')
                    efields = sim_station.get_electric_fields()
                    for efield in efields:
                        # plot simulated electric fields
                        fig1, (ax1_a, ax1_b) = plt.subplots(1, 2, figsize=(10, 6))                        
                        # stored are r, theta, phi components of the electric field, plot only theta and phi
                        for iter in [1, 2]:
                            ax1_a.plot(efield.get_times(), efield.get_trace()[iter], label=label[iter], linestyle=linestyle[iter], linewidth= 2)
                            ax1_b.plot(efield.get_frequencies()/units.MHz, np.abs(efield.get_frequency_spectrum()[iter]), label=label[iter], linestyle=linestyle[iter], linewidth= 2)
                        ax1_a.set_xlabel('Time [ns]', fontsize=18)
                        ax1_a.set_ylabel('Electric field [V/m]', fontsize=18)
                        ax1_a.legend(fontsize= 18)
                        ax1_a.tick_params(axis='x', labelsize= 16)
                        ax1_a.tick_params(axis='y', labelsize=16)
                        ax1_b.set_xlabel('Frequency [MHz]', fontsize=18)
                        ax1_b.set_ylabel('Electric field [V/m/GHz]', fontsize=18)
                        ax1_b.set_xlim(0, 500)
                        ax1_b.legend(fontsize= 18)
                        ax1_b.tick_params(axis='x', labelsize= 16)
                        ax1_b.tick_params(axis='y', labelsize=16)
                        fig1.suptitle(f'Event {event_id}, Station {sta.get_id()}, distance to core {distance_core:.0f}m')
                        fig1.tight_layout()
                        plt.show()

                    # plot reconstruceted voltage trace of the channels
                    fig2, (ax2_a, ax2_b) = plt.subplots(1, 2, figsize=(10, 6))
                    for channel_id in sta.get_channel_ids():
                        print('channel id', channel_id)
                        channel = sta.get_channel(channel_id)
                        trace = channel.get_trace()
                        trace_time = channel.get_times()
                        trace_freqs = channel.get_frequencies()
                        trace_freq_spec = channel.get_frequency_spectrum()

                        ax2_a.plot(trace_time, trace, label=f'Channel {channel_id}')
                        ax2_b.plot(trace_freqs/units.MHz, np.abs(trace_freq_spec), label=f'Channel {channel_id}')
                    ax2_a.set_xlabel('Time [ns]', fontsize=18)
                    ax2_a.set_ylabel('Amplitude [V]', fontsize=18)
                    ax2_a.legend(fontsize= 18)
                    ax2_a.tick_params(axis='x', labelsize= 16)
                    ax2_a.tick_params(axis='y', labelsize=16)
                    ax2_b.set_xlabel('Frequency [MHz]', fontsize=18)
                    ax2_b.set_ylabel('Amplitude [V/Hz]', fontsize=18)
                    ax2_b.legend(fontsize= 18)
                    ax2_b.tick_params(axis='x', labelsize= 16)
                    ax2_b.tick_params(axis='y', labelsize=16)
                    fig2.suptitle(f'Event {event_id}, Station {sta.get_id()}, distance to core {distance_core:.0f}m')
                    fig2.tight_layout()
                    plt.show()