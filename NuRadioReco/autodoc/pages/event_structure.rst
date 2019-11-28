Data Structure
===========================


.. image:: ../images/event_structure.png
  :width: 80%

Event
____________
  The `Event <../NuRadioReco.framework.html#module-NuRadioReco.framework.event>`_
  is the upper-most element of the event structure and holds all simulated and reconstructed
  showers and stations as well as the event ID and run number.

Radio Shower
____________
  A `Radio Shower <../NuRadioReco.framework.html#module-NuRadioReco.framework.radio_shower>`_ is used to
  hold reconstructed shower parameters via the parameter storage. It should only be
  used for properties reconstructed from the radio signal, for properties from a simulated
  shower or reconstructed from another detector, the SimShower or HybridShower should be
  used, respectrively.

  It can be accessed by the ``get_showers`` and ``get_first_shower`` methods of the `Event <../NuRadioReco.framework.html#NuRadioReco.framework.event.Event>`_ class.

SimShower
____________
  A Sim Shower is used to hold parameters of simulated showers via the parameter storage.
  They are the same class as ``RadioShower``, but are stored separately to distinguish
  between simulated and reconstructed properties.

  It can be accessed by the ``get_sim_showers`` method of the `Event <../NuRadioReco.framework.html#NuRadioReco.framework.event.Event>`_ class.

Station
____________
  A `Station <../NuRadioReco.framework.html#module-NuRadioReco.framework.station>`_ is used to hold event properties
  reconstructed at the station level, i.e. reconstructed from the data of a single station.

  It can be accessed by the ``get_station`` and ``get_stations`` methods of the ``Event`` class
Trigger
____________

SimStation
____________
  A `SimStation <../NuRadioReco.framework.html#module-NuRadioReco.framework.sim_station>`_ can hold the same
  properties as the ``Station`` (and inherits from it), but is used for the MC truth  of the simulation. This
  also implies that events from measured data typically do not have a ``SimStation``.

  It can be accessed by the ``get_sim_station`` method of the ``Station`` class.

BaseTrace
____________
  The `BaseTrace <../NuRadioReco.framework.html#module-NuRadioReco.framework.base_trace>`_ class
  is used to store waveforms, both for voltages in the channels and electric fields.
  While internally traces are stored in the time
  domain, where they can be accessed via the ``get_trace`` and ``set_trace`` method, it is also
  possible access the waveform in the frequency domain via the ``get_frequency_spectrum``
  and ``set_frequency_spectrum`` method. In that case, a Fourier transformation is
  done automatically by the ``Trace``.
  The times and frequencies corresponding to the waveforms returned by the ``get_trace``
  and ``get_frequency_spectrum`` methods can be accessed via the ``get_times`` and
  ``get_frequencies`` methods. The times are defined relative to the time
  of the parent ``Station`` and can be changes using the ``set_trace_start_time``
  method, which changes the starting time of the trace.


  The ``Trace`` class is not used by itself, but serves as parent class for both
  the ``Channel`` and ``ElectricField`` classes.

Electric Field
____________
  The `ElectricField <../NuRadioReco.framework.html#module-NuRadioReco.framework.electric_field>`_
  is used to store information about electric fields, which can be accessed via the parameter storage
  and methods inherited from the ``BaseTrace`` class.

  Since radio stations for neutrino detection are often so spread out that the electric field
  is not the same at all channels, each electric field is associated with one or more channels,
  whose IDs have to be passed to the Constructor function and can be accessed by the ``get_channel_ids``
  method. Since pulses may reach a channel via different paths through the ice, multiple ``ElectricField``
  objects may be associated with the same channel. Since typically multiple channels are used to
  reconstruct the electric field, each ``ElectricField`` can be associated with multiple channels. To
  avoid ambiguity, the ``ElectricField`` also has a position (accessed via ``get_position``) relative to
  the station.

  A ``Station`` ´s or ``SimStation`` ´s ``ElectricField`` objects can be accessed via the ``get_electric_fields``
  method or the ``get_electric_fields_for_channels`` method, which allows to filter by channel IDs and ray path types.

Channel
____________
  The `Channel <../NuRadioReco.framework.html#module-NuRadioReco.framework.channel>`_
  is used to store information about the voltage traces recorded in a channel,
  which can be accessed via the parameter storage and methods inherited from
  the ``BaseTrace`` class.


Hybrid Information
____________
  As many radio detectors are built as part of a hybrid detector whose data may be used in the
  radio event reconstruction, a way to make this data accessible in NuRadioReco is needed. The
  `HybridInformation <../NuRadioReco.framework.html#module-NuRadioReco.framework.hybrid_information>`_
  class provides this functionality and sections the information from the
  other detectors off from the radio part to avoid confusion. Despite its name, it does not
  hold any data from the other detectors itself, but offers access to ``HybridShower`` objects in
  which this data is stored. For each additional detector (or set of detector data), a ``HybridSHower``
  object can be added via the ``add_hybrid_shower`` method or accessed via the ``get_hybrid_shower``
  or ``get_hybrid_showers`` methods.

  It can be accessed via the ``get_hybrid_information'' method of the ``Event`` class.

Hybrid Shower
____________
  The `HybridShower <../NuRadioReco.framework.html#module-NuRadioReco.framework.hybrid_shower>`_ is
  used to store information about a shower that was reconstructed with a complementary detector,
  mainly via the parameter storage.

  It can be accessed via the ``get_hybrid_shower`` and ``get_hybrid_showers`` methods of the
  ``HybridInformation`` class.

Hybrid Detector
____________
  A ``HybridDetector`` can be used to store more detailed and experiment-specific information
  about a complementary detector. The diversity of hybrid radio detectors makes it
  impractical to provide this functionality inside NuRadioReco itself, but a custom
  ``HybridDetector`` class can be impemented inside an independent repository. This class
  can be slotted into the data structure via the ``set_hybrid_detector`` method of the ``HybridShower``
  class and accessed via its ``get_hybrid_detector`` method.

  A ``HybridDetector`` class is required to have a constructor that does not accept any parameters as
  well as a ``serialize`` and a ``deserialize`` function equivalent to the other framework elements.

  An example for the implementation of a custom ``HybridDetector`` can be found in the
  NuRadioReco/example folder.
