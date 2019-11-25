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

  It can be accessed by the ``get_sim_station`` method of the ``Station` class.`

Electric Field
____________
  The `ElectricField <../NuRadioReco.framework.html#module-NuRadioReco.framework.electric_field>`_
Channel
____________
Trace
____________
Hybrid Information
____________
Hybrid Shower
____________
Hybrid Detector
____________
