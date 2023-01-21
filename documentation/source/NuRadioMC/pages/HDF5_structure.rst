HDF5 output structure
=====================

The output of a NuRadioMC simulation is saved in the HDF5 file format, as well as (optionally) in ``.nur`` files.
The data structure of ``.nur`` files is explained :doc:`here </NuRadioReco/pages/event_structure>`.
This page outlines the structure of the HDF5 files v3.0. Find the structure of v2.2 :doc:`v2.2 </NuRadioMC/pages/HDF5_structures_history/HDF5_v2.2.rst>`.


Opening the HDF5 file
---------------------
The HDF5 file can be opened using the ``h5py`` module:

.. code-block:: Python

    import h5py

    f = h5py.File("/path/to/hdf5_file", mode='r')
    attributes = f.attrs

    (...)
    f.close()

If you have many HDF5 files, for example because you ran a simulation parallelized over multiple energy bins,
NuRadioMC contains a convenience function to correctly merge these files -
see :ref:`here <NuRadioMC/pages/Manuals/running_on_a_cluster:4. Merge individual hdf5 output files>` for instructions.

What's behind the HDF5 files
----------------------------
The hdf5 file is created in NuRadioMC/simulation/simulation.py A list of vertices with different arrival direction
(zenith and azimuth) and energy is provided by the event generator. Starting from the vertex, several sub-showers are
created along the track. These are not simulated, but the electric field per sub-shower is provided. Sub-showers that
happen within a certain time interval arrive at the antenna simultaneous and interfere constructively, therefore,
they are summed up.

The ``event_group_id`` is the same for all showers that follow the same first interaction.
The ``shower_id`` is unique for every shower. Shower which interfere constructively are combined into one event and have
the same ``event_id`` starting from 0.

  .. image:: event_sketch.png
    :width: 70%

HDF5 structure
--------------
The HDF5 files can be thought of as a structured dictionary:

- The top level :ref:`attributes <NuRadioMC/pages/HDF5_structure:HDF5 file attributes>`, which can be accessed through ``f.attrs``, contain some top-level information about the simulation.
- The :ref:`individual keys <NuRadioMC/pages/HDF5_structure:HDF5 file contents>` contain some properties (energy, vertex, ...) for each stored event or shower.
- Finally, the ``station_<station_id>`` key contains slightly more detailed information (triggers, propagation times, amplitudes...) at the level of individual channels :ref:`for each station <NuRadioMC/pages/HDF5_structure:Station data>`.

HDF5 file attributes
____________________

The top-level attributes can be accessed using ``f.attrs``. These contain:

    .. _hdf5-attrs-table:

    .. csv-table:: HDF5 attributes
            :header: "Key", "Description"
            :widths: auto
            :delim: |

            ``NuRadioMC_EvtGen_version`` ``NuRadioMC_EvtGen_version_hash`` | Hashes
            ``NuRadioMC_version`` ``NuRadioMC_version_hash`` | Hashes
            ``Emin`` ``Emax`` | Define energy range for neutrino energies
            ``phimax`` ``phimin`` | Define azimuth range for incoming neutrino directions
            ``thetamax`` ``thetamin`` | Define zenith range for incoming neutrino directions
            ``flavors`` | A list of particle flavors that were simulated, using the PDG convention.
            ``n_events`` | Total number of generated/simulated events(including those that did not trigger)
            ``fiducial_xmax`` ``fiducial_xmin`` ``fiducial_ymax`` ``fiducial_ymin`` ``fiducial_zmax`` ``fiducial_zmin`` / ``fiducial_rmax`` ``fiducial_rmin`` ``fiducial_zmax`` ``fiducial_zmin`` | Specify the simulated qubic/cylindrical fiducial volume.  An event has to produce an interaction within this volume. However, in case of a muon or tau CC interaction the first interaction can occur outside
            ``rmax`` ``rmin`` ``zmax`` ``zmin`` / ``xmax`` ``xmin`` ``ymax`` ``ymin`` ``zmax`` ``zmin`` | Specify the qubic/cylindrical volume in which neutrino interactions are generated
            ``volume`` | Volume of the above specified volume
            ``area`` | Surface area of the above specified volume
            ``start_event_id`` | ``event_id`` of the first event in the file
            ``trigger_names`` | List of the names of the different triggers simulated
            ``Tnoise`` | (explicit) noise temperature used in simulation
            ``Vrms`` |
            ``bandwidth`` |
            ``n_samples`` |
            ``config`` | The (yaml-style) config file used for the simulation
            ``deposited`` |
            ``detector`` | The (json-format) detector description used for the simulation
            ``dt`` | The time resolution, i.e. the inverse of the sampling rate used for the simulation. This is not necessarily the same as the sampling rate of the simulated channels!

HDF5 file contents
__________________
The HDF5 file contains the following items. Listed are the ``key`` and the ``shape`` of each HDF5 dataset, where ``n_events`` is the number of events stored in the file and ``n_showers``
is the number of showers (which may be larger than the number of events), and ``n_triggers`` is the number of different triggers simulated. Each "row" correspond to a particle shower which can produce radio emission.

    .. _hdf5-items-table:

    .. csv-table:: HDF5 items
            :header: "Key", "Shape", "Description"
            :widths: auto
            :delim: |

            ``event_group_ids`` | (``n_showers``) | Specifies the event id to which the corresponding shower belongs (``n_events = len(unique(event_group_ids)))``)
            ``xx`` ``yy`` ``zz`` | (``n_showers``) | Specifying coordinates of interaction vertices
            ``vertex_times`` | (``n_showers``) | Time at the interaction vertex. The neutrino interaction (= first interaction) is defined as time 0
            ``azimuths`` ``zeniths`` | (``n_showers``) | Angle Specifying the neutrino incoming direction (``azimuths = 0`` points east)
            ``energies`` | (``n_showers``) | Energy of the parent particle of a shower. This is typically the energy of the neutrino (for showers produced at the first interaction: all flavor NC, electron CC interactions) or the energy of a muon or tau lepton when those are producing secondary energy losses
            ``shower_energies`` | (``n_showers``) | Energy of the shower which is used to determine the radio emission
            ``flavors`` | (``n_showers``) | Same as above (the parent of an electromagnetic cascade in an electron CC interaction is the neutrino)
            ``inelasticity`` | (``n_showers``) | Inelasticity of the first interaction
            ``interaction_type`` | (``n_showers``) | Interaction type producing the shower (for the first interaction that can be "nc" or "cc")
            ``multiple_triggers`` | (``n_showers``, ``n_triggers``) | Information which exact trigger fired each shower. The different triggers are specified in the attributes (``f.attrs["triggers"]``). The order of ``f.attrs["triggers"]`` matches that in ``multiple_triggers``
            ``triggered`` | (``n_showers``) | A boolean; ``True`` if any trigger fired for this shower, ``False`` otherwise
            ``n_interaction`` | (``n_showers``) | Hierarchical counter for the number of showers per event (also accounts for showers which did not trigger and might not be saved)
            ``shower_ids`` | (``n_showers``) | Hierarchical counter for the number of triggered showers
            ``shower_realization_ARZ`` | (``n_showers``) | Which realization from the ARZ shower library was used for each shower (only if ARZ was used for signal generation).
            ``shower_type`` | (``n_showers``) | Type of the shower (so far we only have "em" and "had")
            ``weights`` | (``n_showers``) | Weight for the probability that the neutrino reached the interaction vertex taking into account the attenuation from the earth (Does not include interaction probability in the volume)


Station data
____________
In addition, the HDF5 file contains a key for each station in the simulation.
The station contains more detailed information for each event that triggered it:
``m_events`` and ``m_showers`` refer to the number of events and showers that triggered the station.
The ``event_group_id`` is the same as in the global dictionary. Therefore you can check for one event with
an ``event_group_id`` which stations contain the same ``event_group_id`` and retrieve the information, which
station triggered, with which amplitude, etc. The same approach works for ``shower_id``.

    .. _hdf5-station-table:

    .. csv-table:: HDF5 station items
            :header: "Key", "Shape", "Description"
            :widths: auto
            :delim: |

            ``event_group_ids`` | (``m_events``) | The event group ids of the triggered events in the selected station
            ``event_group_id_per_shower`` | (``m_showers``) | The event group id of every shower that triggered the selected station
            ``event_ids`` | (``m_events``) | The event ids of each event that triggered in that station for every event group id. These are unique only within each separate event group, and start from 0.
            ``event_id_per_shower`` | (``m_showers``) | The event ids of each event that triggered in that station. This one is for every shower
            ``focusing_factor`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``) |
            ``launch_vectors`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``, ``3``) | 3D (Cartesian) coordinates of the launch vector of each ray tracing solution, per shower and channel.
            ``max_amp_shower_and_ray`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``) | Maximum amplitude per shower, channel and ray tracing solution.
            ``maximum_amplitudes`` | (``m_events``, ``n_channels``) | Maximum amplitude per event and channel
            ``maximum_amplitudes_envelope`` | (``m_events``, ``n_channels``) | Maximum amplitude of the hilbert envelope for each event and channel
            ``multiple_triggers`` | (``m_showers``, ``n_triggers``) | A boolean array that specifies if a shower contributed to an event that fulfills a certain trigger. The index of the trigger can be translated to the trigger name via the attribute ``trigger_names``.
            ``multiple_triggers_per_event`` | (``m_events``, ``n_triggers``) | A boolean array that specifies if each event fulfilled a certain trigger. The index of the trigger can be translated to the trigger name via the attribute ``trigger_names``.
            ``polarization`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``, ``3``) | 3D (Cartesian) coordinates of the polarization vector
            ``ray_tracing_C0`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``) | One of two parameters specifying the **analytic** ray tracing solution. Can be used to retrieve the solutions without having to re-run the ray tracer.
            ``ray_tracing_C1`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``) | One of two parameters specifying the **analytic** ray tracing solution. Can be used to retrieve the solutions without having to re-run the ray tracer.
            ``ray_tracing_reflection`` | (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``) |
            ``ray_tracing_reflection_case`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``) |
            ``ray_tracing_solution_type`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``) |
            ``receive_vectors`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``, ``3``) | 3D (Cartesian) coordinates of the receive vector of each ray tracing solution, per shower and channel.
            ``shower_id`` | (``m_showers``) | The Shower ids of showers that triggered the selected station
            ``time_shower_and_ray`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``) |
            ``travel_distances`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``) | The distance travelled by each ray tracing solution to a specific channel
            ``travel_times`` | (``m_showers``, ``n_channels``, ``n_ray_tracing_solutions``) | The time travelled by each ray tracing solution to a specific channel
            ``triggered`` | (``m_showers``) | Whether each shower contributed to an event that satisfied any trigger condition
            ``triggered_per_event`` | (``m_events``) | Whether each event fulfilled any trigger condition.
