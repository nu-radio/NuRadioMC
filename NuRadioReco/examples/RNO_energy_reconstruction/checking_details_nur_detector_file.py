import NuRadioReco.modules.io.eventReader as eventReader
import NuRadioReco.detector.detector as detector
from NuRadioReco.framework.parameters import eventParameters as ep

from NuRadioReco.framework.parameters import channelParameters as chp

from NuRadioReco.framework.parameters import showerParameters as sh

from NuRadioReco.framework.parameters import stationParameters as st

from NuRadioReco.framework.parameters import particleParameters as pp

# Load the detector (if not embedded)
det = detector.Detector(json_filename="RNO_single_station.json")
# Open the NuRadioReco file
reader = eventReader.eventReader()
reader.begin("simulated_events.nur")

n_events = 0
n_triggered = 0

for event in reader.run():
    n_events += 1
    station = event.get_station(11)  # station ID 11

    if station.has_triggered():
        n_triggered += 1

print("Total events in file:", n_events)
print("Number of events triggered:", n_triggered)
# print("Trigger fraction:", n_triggered / n_events)

for event in reader.run():
    event.show(show_parameters=0)
    station = event.get_station(11)
    nu_vert = station.get_parameter(st.nu_vertex)
    print("Neutrino vertex (x, y, z) [m]:", nu_vert)
    


    # parti = event.get_particle()  # neutrino has particle ID 1
    # print(parti.get_parameters())
    # # sim_station = event.get_station(11).get_sim_station()
    # # print("Number of sim showers:", len(sim_showers))

    # print("Station neutrino")                           #no one has set it  i want to see the sim_station

    # z = sim_station.get_parameter(st.zenith)
    # print(z)
    # print("neutrino zenith above")



    # for sim_shower in event.get_sim_showers():
    #     z = sim_shower.get_parameter(sh.zenith)
    #     # x = sim_shower.get_parameter(sh.zenith)
    #     # y = sim_shower.get_parameter(sh.zenith)
    #     # t = sim_shower.get_parameter(sh.zenith)
    #     print("zenith")
    #     print(zenith)
    #     # y = sim_shower.get_parameter(sim_shower.parameters.vertex_y)
        # z = sim_shower.get_parameter(sim_shower.parameters.vertex_z)
        # t = sim_shower.get_parameter(sim_shower.parameters.vertex_time)

        # print(f"Shower vertex:")
        # print(f"  x = {x:.2f} m")
        # print(f"  y = {y:.2f} m")
        # print(f"  z = {z:.2f} m")
        # print(f"  t = {t:.2e} s")
    # zenith_true = event.get_parameter(ep.zenith)
    # azimuth_true = event.get_parameter(ep.azimuth)
    # print(f"Neutrino direction: zenith={zenith_true:.3f}, azimuth={azimuth_true:.3f}")
    # break  # just first event for quick check



# reader = eventReader.eventReader()
# reader.begin("1e19_n100_output.nur")

# for event in reader.run():
#     station = event.get_station(11)

#     for ch in range(station.get_num_channels()):
#         max_voltage = station.get_channel(ch).get_parameter(chp.v_peak)
#         print(f"Channel {ch} peak voltage: {max_voltage:.2f} mV")
#     break
