HDF5 output structure
=====================

The output of a NuRadioMC simulation is saved in the HDF5 file format, as well as (optionally) in ``.nur`` files.
The data structure of ``.nur`` files is explained :doc:`here </NuRadioReco/pages/event_structure>`.
This page outlines the structure of the HDF5 files.

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

HDF5 structure
--------------
The HDF5 files can be thought of as a structured dictionary:

- The top level :ref:`attributes <NuRadioMC/pages/HDF5_structure:HDF5 file attributes>`, which can be accessed through ``f.attrs``, contain some top-level information about the simulation.
- The :ref:`individual keys <NuRadioMC/pages/HDF5_structure:HDF5 file contents>` contain some properties (energy, vertex, ...) for each stored event or shower.
- Finally, the ``station_<station_id>`` key contains slightly more detailed information (triggers, propagation times, amplitudes...) at the level of individual channels :ref:`for each station <NuRadioMC/pages/HDF5_structure:Station data>`.

HDF5 file attributes
____________________
The top-level attributes can be accessed using ``f.attrs``. These contain:

* ``Emax``, ``Emin``

  maximum and minimum energy simulated
* ``NuRadioMC_EvtGen_version``, ``NuRadioMC_EvtGen_version_hash``
* ``NuRadioMC_version``, ``NuRadioMC_version_hash``
* ``Tnoise``

  (explicit) noise temperature used in simulation
* ``Vrms``
* ``area``
* ``bandwidth``
* ``config``

  the (yaml-style) config file used for the simulation
* ``deposited``
* ``detector``

  the (json-format) detector description used for the simulation
* ``dt``

  the time resolution, i.e. the inverse of the sampling rate used for the simulation.
  This is not necessarily the same as the sampling rate of the simulated channels!
* ``fiducial_rmax``, ``fiducial_rmin``, ``fiducial_zmax``, ``fiducial_zmin``

  Specify the simulated fiducial volume
* ``flavors``

  a list of particle flavors that were simulated, using the PDG convention.
* ``n_events``

  total number of events simulated (including those that did not trigger)
* ``n_samples``
* ``phimax``, ``phimin``
* ``rmax``, ``rmin``
* ``start_event_id``

  ``event_id`` of the first event in the file
* ``thetamax``, ``thetamin``
* ``trigger_names``

  list of the names of the different triggers simulated
* ``volume``
* ``zmax``, ``zmin``

HDF5 file contents
__________________
The HDF5 file contains the following items. Listed are the ``key`` and the ``shape`` of
each HDF5 dataset, where ``n_events`` is the number of events in the file, ``n_showers``
is the number of showers (which may be larger than the number of events), and ``n_triggers``
is the number of different triggers simulated.

* ``azimuths``: (``n_events``,)
* ``energies``: (``n_events``,)
* ``event_group_ids``: (``n_events``,)
* ``flavors``: (``n_events``,)
* ``inelasticity``: (``n_events``,)
* ``interaction_type``: (``n_events``,)
* ``multiple_triggers``: (``n_events``, ``n_triggers``)
* ``n_interaction``: (``n_events``,)
* ``shower_energies``: (``n_showers``,)
* ``shower_ids``: (``n_showers``,)
* ``shower_realization_ARZ``: (``n_showers``,)

  Which realization from the ARZ shower library was used for each shower (only if ARZ
  was used for signal generation).
* ``shower_type``: (``n_showers``,)
* ``triggered``: (``n_events``,)

  boolean; ``True`` if the event triggered on any trigger, ``False`` otherwise
* ``vertex_times``: (``n_events``,)
* ``weights``: (``n_events``,)
* ``xx``: (``n_events``,)
* ``yy``: (``n_events``,)
* ``zeniths``: (``n_events``,)
* ``zz``: (``n_events``,)

Station data
____________
In addition, the HDF5 file contains a key for each station in the simulation.
The station contains more detailed information for each event that triggered it:

* ``event_group_ids``: (``n_events``,)

  event group ids of the triggered events
* ``event_ids``: (``n_events``,)

  the event ids of each event. These are unique only within each separate event group,
  and start from 0.
* ``focusing_factor``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``)
* ``launch_vectors``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``, 3)

  3D (Cartesian) coordinates of the launch vector of each ray tracing solution,
  per shower and channel.
* ``max_amp_shower_and_ray``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``)

  Maximum amplitude per shower, channel and ray tracing solution.
* ``maximum_amplitudes``: (``n_events``, ``n_channels``)

  Maximum amplitude per event and channel
* ``maximum_amplitudes_envelope``: (``n_events``, ``n_channels``)

  Maximum amplitude of the hilbert envelope for each event and channel
* ``multiple_triggers``: (``n_showers``, ``n_triggers``)

  a boolean array that specifies if a shower contributed to an event that fulfills a certain trigger.
  The index of the trigger can be translated to the trigger name via the attribute ``trigger_names``.
* ``multiple_triggers_per_event``: (``n_events``, ``n_triggers``)

  a boolean array that specifies if each event fulfilled a certain trigger.
  The index of the trigger can be translated to the trigger name via the attribute ``trigger_names``.
* ``polarization``: (``n_shower``, ``n_channels``, ``n_ray_tracing_solutions``, 3)

  3D (Cartesian) coordinates of the polarization vector
* ``ray_tracing_C0``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``)

  One of two parameters specifying the **analytic** ray tracing solution.
  Can be used to retrieve the solutions without having to re-run the ray tracer.
* ``ray_tracing_C1``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``)

  One of two parameters specifying the **analytic** ray tracing solution.
  Can be used to retrieve the solutions without having to re-run the ray tracer.
* ``ray_tracing_reflection``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``)
* ``ray_tracing_reflection_case``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``)
* ``ray_tracing_solution_type``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``)
* ``receive_vectors``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``, 3)

  3D (Cartesian) coordinates of the receive vector of each ray tracing solution,
  per shower and channel.
* ``shower_id``: (``n_showers``,)
* ``time_shower_and_ray``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``)
* ``travel_distances``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``)

  The distance travelled by each ray tracing solution to a specific channel
* ``travel_times``: (``n_showers``, ``n_channels``, ``n_ray_tracing_solutions``)

  The time travelled by each ray tracing solution to a specific channel
* ``triggered``: (``n_showers``,)

  Whether or not each shower contributed to an event that satisfied any trigger condition
* ``triggered_per_event``: (``n_events``,)

  Whether or not each event fulfilled any trigger condition.
