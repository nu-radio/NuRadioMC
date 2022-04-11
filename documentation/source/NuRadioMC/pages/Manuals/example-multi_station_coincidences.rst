Example: Multi-station coincidences
===================================

In this example we calculate the probability to measure the same neutrino in multiple stations. 
For a discovery detector, one objective is a large sensitivity which means that stations 
need to be separated far enough to minimize station coincidences. 
Here, we show how this can be studied as a function of station separation and antenna depth with NuRadioMC.

1. Generation of detector layout
---------------------------------------
We consider two simplified detectors. The first one is a surface oriented station consisting of LPDAs and dipoles. 
To save computing time, we only simulate two orthogonally oriented horizontal LPDAs at 2m depth and one dipole at 5m depth 
to cover all signal polarizations. The second one is a deep detector, approximated with a single dipole antenna at 50m depth. 
We combine the four antennas into a single station:

.. code-block:: json

    {
        "channels": {
            "1": {
                "adc_n_samples": 512,
                "adc_sampling_frequency": 1.0,
                "ant_orientation_phi": 0.0,
                "ant_orientation_theta": 180.0,
                "ant_position_x": -3.0,
                "ant_position_y": -0.0,
                "ant_position_z": -1.0,
                "ant_rotation_phi": 0.0,
                "ant_rotation_theta": 90.0,
                "ant_type": "createLPDA_100MHz",
                "channel_id": 0,
                "commission_time": "{TinyDate}:2017-11-01T00:00:00",
                "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
                "station_id": 101
            },
            "2": {
                "adc_n_samples": 512,
                "adc_sampling_frequency": 1.0,
                "ant_orientation_phi": 0.0,
                "ant_orientation_theta": 180.0,
                "ant_position_x": 0.0,
                "ant_position_y": 3.0,
                "ant_position_z": -1.0,
                "ant_rotation_phi": 270.0,
                "ant_rotation_theta": 90.0,
                "ant_type": "createLPDA_100MHz",
                "channel_id": 1,
                "commission_time": "{TinyDate}:2017-11-01T00:00:00",
                "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
                "station_id": 101
            },
            "3": {
                "adc_n_samples": 512,
                "adc_sampling_frequency": 1.0,
                "ant_orientation_phi": 0.0,
                "ant_orientation_theta": 0.0,
                "ant_position_x": 3.0,
                "ant_position_y": 3.0,
                "ant_position_z": -5.0,
                "ant_rotation_phi": 90.0,
                "ant_rotation_theta": 90.0,
                "ant_type": "bicone_v8",
                "channel_id": 2,
                "commission_time": "{TinyDate}:2017-11-01T00:00:00",
                "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
                "station_id": 101
            },
            "4": {
                "adc_n_samples": 512,
                "adc_sampling_frequency": 1.0,
                "ant_orientation_phi": 0.0,
                "ant_orientation_theta": 0.0,
                "ant_position_x": 0,
                "ant_position_y": 0,
                "ant_position_z": -40.0,
                "ant_rotation_phi": 90.0,
                "ant_rotation_theta": 90.0,
                "ant_type": "bicone_v8",
                "channel_id": 3,
                "commission_time": "{TinyDate}:2017-11-01T00:00:00",
                "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
                "station_id": 101
            }
        },
        "stations": {
            "1": {
                "commission_time": "{TinyDate}:2017-11-04T00:00:00",
                "decommission_time": "{TinyDate}:2038-01-01T00:00:00",
                "pos_altitude": 2800.0,
                "pos_site": "southpole",
                "position": "SP1",
                "station_id": 101
            }
        }
    }

and then use a Python script to generate a x-y grid of stations. 
In principle, we would have needed to simulate a full 2D grid 
for every station separation distance that we wanted to test, 
because there might be cases where not the nearest neighbors triggered 
but the next-to nearest neighbors or stations even further out. 
However, as this will strongly increase computing time 
(which scales linearly with the number of channels) we ignore this small second order effect.
Our analysis will show that the coincidence rate will drop quickly if the separation between stations is doubled.
Hence, the coincidence rate is dominated by the nearest neighbors which justifies our approximation. 
Hence, for every station separation distance, we place consider only the eight nearest stations around the central station as illustrated in Fig.?. 

The following scripts generates the json detector description.

.. code-block:: Python

    import copy
    import json

    with open("single_position.json") as fin:
        detector_single = json.load(fin)
        
        detector_full ={}
        detector_full['stations'] = detector_single['stations'] 
        detector_full['channels'] = {}
        # insert station at center
        i = -1
        for channel in detector_single['channels'].values():
            i += 1
            channel = copy.copy(channel)
            channel['channel_id'] = i
            detector_full['channels'][str(i+1)] = channel
        
        distances = [100, 250, 500, 750, 1000, 1250, 1500]
        xx = [0]
        yy = [0]
        for d in distances:
            for x in [-d, 0, d]:
                for y in [-d, 0, d]:
                    if(x == 0 and y == 0):
                        continue
                    for channel in detector_single['channels'].values():
                        i += 1
                        channel = copy.copy(channel)
                        channel['ant_position_x'] += (x)
                        channel['ant_position_y'] += (y)
                        channel['channel_id'] = i
                        detector_full['channels'][str(i+1)] = channel
                        xx.append(x)
                        yy.append(y)
                
        with open('horizontal_spacing_detector.json', 'w') as fout:
            json.dump(detector_full, fout, indent=4, separators=(',', ': '))

Simulating such a detector will allow to determine the probability that the central station at (0,0) 
measured a signal in coincidence with any of its surrounding stations of a certain distance. 

2. Detector simulation
-----------------------
We define the typical detector simulation, apply a bandpass filter from 80 - 500 MHz to mimic a typical amplifier response,
and run a simple threshold trigger that saves all events where one channel has a signal above 1xVrms.
This trigger is chosen to obtain a rough preselection that could fulfill the coincidence criterion. 

.. code-block:: Python

    def detector_simulation(evt, station, det, dt, Vrms):
        # start detector simulation
        efieldToVoltageConverterPerChannel.run(evt, station, det)  # convolve efield with antenna pattern
        # downsample trace back to detector sampling rate
        channelResampler.run(evt, station, det, sampling_rate=1. / dt)
        # bandpass filter trace, the upper bound is higher then the sampling rate which makes it just a highpass filter
        channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 500 * units.MHz],
                                filter_type='butter10')
        triggerSimulator.run(evt, station, det,
                            threshold=1 * Vrms,
                            triggered_channels=None,
                            number_concidences=1,
                            trigger_name='pre_trigger_1sigma'

3. Running the simulation
--------------------------
The next step is to generate the input event lists and to run the simulation. 
As we study the multi-station coincidences for different neutrino energies, we generate separate input event lists per fixed neutrino energy as described in the :doc:`Effective volume tutorial </NuRadioMC/pages/Manuals/veff_tutorial>`. 

If many similar studies are performed, a significant gain in computing time can be achieved with the following trick: We first run a simulation with a single station at (x,y) = (0,0) and a low trigger threshold. Then, we use the output of the pre-simulation as the input of this study. Hence, we only need to simulate a reduced event data set, and don't spend a lot of computing time on simulating events that don't trigger the central station. 


4. Analyzing the output
-----------------------
The NuRadioMC output is available in one hdf5 file per simulated neutrino energy. (If the data set was split up in small files to run it simultaneously on a cluster, the hdf5 needs to be merged first as describe :ref:`here <NuRadioMC/pages/Manuals/running_on_a_cluster:4. Merge individual hdf5 output files>`). 

Part of the output file is the maximum amplitude of each channel of each event stored in a two dimensional array. This allows a quick calculation of the coincidence requirements. We first check if the central station fulfilled the trigger condition which we assume to be a signal above 3xVrms in any channel. Then, for each simulated distance, we select the channels corresponding to this distance and check if any channel fulfills the trigger condition. The coincidence rate is then given by the ratio of events where both the central station and any of its nearest neighbors triggered, divided by the number of triggers of the central station alone. 

This calculation is done with the following Python code

.. code-block:: Python

    ...
    fig, ax = plt.subplots(1, 1)
    for iF, filename in enumerate(sorted(glob.glob("*.hdf5"))):

        fin = h5py.File(filename)
        with open('det.json', 'w') as fout:
            fout.write(fin.attrs['detector'])
            fout.close()
        det = detector.Detector(json_filename="det.json")
        max_amps_env = np.array(fin['maximum_amplitudes_envelope'])

        xs = np.zeros(max_amps_env.shape[0])
        ys = np.zeros(max_amps_env.shape[0])
        coincidence_fractions = np.zeros(max_amps_env.shape[0])

        triggered_near = np.any(max_amps_env[:, 0:3] > (3 * Vrms), axis=1)  # triggered any LPDA or dipole of the center station
        triggered = np.zeros((max_amps_env.shape[0], max_amps_env.shape[1] / 4)) # create empy array of shape (n events, n stations) (we had 4 antennas per station)
        # loop through all stations with different distances (which is just the total number of channels divided by 4)
        for i in range(det.get_number_of_channels(101) / 4):
            # select the 2 LPDA + 1 dipole channel and check if they fulfill the trigger condition
            triggered[:, i] = np.any(max_amps_env[:, i * 4:(i * 4 + 3)] > (3 * Vrms), axis=1)
            # get their position
            xs[i] = np.abs(det.get_relative_position(101, i * 4)[0])
            ys[i] = np.abs(det.get_relative_position(101, i * 4)[1])
        # loop through all simulated distances
        for i, x in enumerate(np.unique(xs)):
            mask = (xs == x) & (ys == x)  # select all stations corresponding to this distance
            # calculate coincidence fraction
            coincidence_fractions[i] = 1. * np.sum(np.any(triggered[:, mask], axis=1) & triggered_near) / np.sum(triggered_near)

        ax.plot(np.unique(xs), coincidence_fractions, php.get_marker2(iF)+'-', label="E = {:.2g}".format(fin.attrs['Emin']))
    ax.set_xlabel("distance [m]")
    ax.set_ylabel("coincidence fraction")
    ax.set_title("surface antennas")
    ax.semilogy(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("coincidences.png")
    plt.show()


