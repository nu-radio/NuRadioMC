Data Structure
===========================

.. Note::

  This page explains the structure of NuRadioReco ``.nur`` files, which contain event-level data
  from simulation or reconstruction. For more information on the simpler ``.hdf5`` files produced
  in a ``NuRadioMC`` simulation, see the page on :doc:`HDF5 structure </NuRadioMC/pages/HDF5_structure>`.

.nur Files and How to Use Them
----------------------------------

Philosophy and Basic Structure
__________________________________
NuRadioReco comes with its own input and output format, called *.nur*. With
the obvious exception of reading in data from other file formats, like
*CoREAS* or the *SnowShovel* format of the *ARIANNA* experiment, every
processing step in an event reconstruction is done in this format. The big
advantage this provides, is that at any point the process can be interrupted
and the current state of the event data can be saved. This makes it easy to
split a reconstruction into several steps and to check the state of the data
structure after every step.

.. image:: event_structure.png
  :width: 100%

A NuRadioReco event is organized hierarchical, with an ``Event`` object at the
top. Elements further down the hierarchy can be accessed via *get* functions or
iterators from their parent object. For example, accessing the traces of a
station's channels would work like this:

.. code-block:: Python

  #get station with ID 42
  station = event.get_station(42)
  # iterate over all channels in station
  for channel in station.iter_channels():
    trace = channel.get_trace()

Reading and Writing .nur Files
________________________________

Reading and writing *.nur* files is done by dedicated IO modules.
Writing events is done by the eventWriter module. To save disk space it offers
the option to not store channel and electric field traces, in case only
the higher-level parameters are needed. It is also possible to :ref:`write the
detector description onto a *.nur* file <NuRadioReco/pages/detector/detector:Writing the detector>`.

.. code-block:: Python

  import NuRadioReco.modules.io.eventWriter

  event_writer = NuRadioReco.modules.io.eventWriter.eventWriter()
  event_writer.begin('output_filename.nur')
  event_writer.run(event, mode='full')

To read *.nur* files, two different modules can be used: :class:`NuRadioRecoio <NuRadioReco.modules.io.NuRadioRecoio>` is a
general-purpose reader that provides different ways to access events e.g. by
ID or by event number. The :class:`eventReader <NuRadioReco.modules.io.eventReader>` is a more streamlined wrapper around
:class:`NuRadioRecoio <NuRadioReco.modules.io.NuRadioRecoio>` that provides an iterator over all events. Both modules provide
as way to :ref:`read the detector description from a *.nur* file <NuRadioReco/pages/detector/detector:Reading the detector>`.

.. code-block:: Python

  import NuRadioReco.modules.io.NuRadioRecoio
  nuradioreco_io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(['path/to/file', '/path/to/other/file'])
  # get event with run number 0 and event ID 5
  event_1 = nuradioreco_io.get_event([0,5])
  # get second event in files (counting starts at 0)
  event_2 = nuradioreco.io.get_event_i(1)
  # iterate over all events
  for event in nuradioreco_io.get_event():
    station = event.get_station(42)

  import NuRadioReco.modules.io.eventReader
  event_reader = NuRadioReco.modules.io.eventReader.eventReader()
  event_reader.begin(['path/to/file', 'path/to/other/file'])
  # iterate over events
  for event in event_reader.run():
    station = event.get_station(42)

Additionally, *.nur* files store higher-level parameters in their headers, which
makes them easily accessible for all events in a file. For example, if one wanted
to make a histogram of the zenith angles in a given file, it would work like this:

.. code-block:: Python

  import matplotlib.pyplot as plt
  from NuRadioReco.framework.parameters import stationParameters as stnp
  from NuRadioReco.utilities import units
  import NuRadioReco.modules.io.NuRadioRecoio
  nuradioreco_io = NuRadioReco.modules.io.NuRadioRecoio.NuRadioRecoio(['path/to/file'])
  header = nuradioreco_io.get_header()
  station_id = 42
  zeniths = header[station_id][stnp.zenith]
  plt.hist(zeniths/units.deg)
  plt.show()

The way that writing and reading *.nur* files is handled internally is that
every class in the framework has a ``serialize`` function that writes all
information stored in the object into a `pickle <https://docs.python.org/3/library/pickle.html>`_ object
and a ``deserialize`` function that writes the data from such a *pickle* into
a class object. To write an event to disk, each object calls the ``serialize``
function on its child objects, stores the *pickles* they return and then
serializes itself. The resulting *pickle* can then be written to disk. To read
a *.nur* file the same is done in reverse, with each object calling the ``deserialize``
function on its children. Thanks to this implementation, it is easy to extend
the framework, since all that has to be done is to define ``serialize`` and
``deserialize`` functions and adjust the ones of the parent object.

Parameter Storage
----------------------
NuRadioReco offers a flexible way to store properties in the data structure via
parameter storage. Certain classes (``Particle``, ``Station``, ``SimStation``, ``Channel``,
``ElectricField``, ``RadioShower`` and ``HybridShower``) provide ``get_parameter``
and ``set_parameter`` functions that allow parameters to be stored in those
objects along with their uncertainties and correlation to any other parameters.
The parameters are defined in an enumerated type *enum*, so to add a new parameter,
it just needs to be added to the
:mod:`list of parameters <NuRadioReco.framework.parameters>`.

.. admonition:: For Developers

  New parameters should always be added to the bottom of the list. Do not re-use old Enums!
  A description should be added to each new parameter with a comment docstring starting with ``#:``.

Additionally, parameters can be written and accessed via indexing, like one
would do to a dictionary:

.. code-block:: Python

  from NuRadioReco.framework.parameters import stationParameters as stnp
  from NuRadioReco.utilities import units

  # both ways to set the parameter are equivalent
  station.set_parameter(stnp.cr_zenith, 45 * units.deg)
  station[stnp.cr_zenith] = 45 * units.deg
  # set parameter uncertainty
  station.set_parameter_error(stnp.cr_zenith, 2 * units.deg)
  # 2 ways of accessing parameters:
  zenith = station.get_parameter(stnp.cr_zenith)
  zenith = station[stnp.cr_zenith]
  # get parameter uncertainty
  zenith_uncertainty = station.get_parameter_error(stnp.cr_zenith)

List of Data Classes
----------------------

Event
____________
The :class:`Event <NuRadioReco.framework.event.Event>`
is the upper-most element of the event structure and holds all simulated and reconstructed
showers and stations as well as the event ID and run number.

Radio Shower
______________
A :class:`Radio Shower <NuRadioReco.framework.radio_shower.RadioShower>` is used to
hold reconstructed shower parameters via the parameter storage. It should only be
used for properties reconstructed from the radio signal, for properties from a simulated
shower or reconstructed from another detector, the SimShower or HybridShower should be
used, respectrively.

It can be accessed by the ``get_showers`` and ``get_first_shower`` methods of the :class:`Event<NuRadioReco.framework.event.Event>` class.

SimShower
____________
A Sim Shower is used to hold parameters of simulated showers via the parameter storage.
They are the same class as ``RadioShower``, but are stored separately to distinguish
between simulated and reconstructed properties.

It can be accessed by the ``get_sim_showers`` method of the :class:`Event<NuRadioReco.framework.event.Event>` class.

SimEmitter
____________
The :class:`SimEmitter<NuRadioReco.framework.sim_emitter.SimEmitter>` class is used to hold parameters of simulated emitters via the parameter storage.
The concept is similar to the SimShower, but is used when NuRadioMC is used to simulate emitters (and)
not particle showers.

It can be accessed by the ``get_sim_emitters`` method of the :class:`Event<NuRadioReco.framework.event.Event>` class.
We allow for multiple emitters per event analogous to the multiple showers per event.

Particle
________
The :class:`Particle<NuRadioReco.framework.particle>` class stores information related to the particle that initiated the radio emission, such as flavour, energy and direction. A single :ref:`Event <NuRadioReco/pages/event_structure:Event>` may contain multiple particles, e.g. in the case of tau regeneration.

Station
____________
A :class:`Station<NuRadioReco.framework.station>` is used to hold event properties
reconstructed at the station level, i.e. reconstructed from the data of a single station.

It can be accessed by the ``get_station`` and ``get_stations`` methods of the ``Event`` class

Trigger
____________
The :class:`Trigger<NuRadioReco.framework.trigger>` contains information about the station trigger - trigger type, threshold, trigger time and whether the trigger condition was satisfied.

SimStation
____________
A :class:`SimStation<NuRadioReco.framework.sim_station>` can hold the same
properties as the ``Station`` (and inherits from it), but is used for the MC truth  of the simulation. This
also implies that events from measured data typically do not have a ``SimStation``.

It can be accessed by the ``get_sim_station`` method of the ``Station`` class.

BaseTrace
____________
The :class:`BaseTrace<NuRadioReco.framework.base_trace>` class
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

The add operator (+) is defined for 2 ``BaseTrace`` objects. It will return a new ``BaseTrace``
object containing the sum of both traces. The length of the new trace is chosen so that
it is long enough to contain both traces. If the traces have different sampling rates,
the one with the lower sampling rate will be upsampled to match the other one.
Since this property is inherited, + is defined for both channels and electric fields.


The ``Trace`` class is not used by itself, but serves as parent class for both
the ``Channel`` and ``ElectricField`` classes.

Electric Field
_______________
The :class:`ElectricField<NuRadioReco.framework.electric_field>`
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
The :class:`Channel<NuRadioReco.framework.channel>`
is used to store information about the voltage traces recorded in a channel,
which can be accessed via the parameter storage and methods inherited from
the ``BaseTrace`` class.


Hybrid Information
___________________
As many radio detectors are built as part of a hybrid detector whose data may be used in the
radio event reconstruction, a way to make this data accessible in NuRadioReco is needed. The
:class:`HybridInformation<NuRadioReco.framework.hybrid_information>`
class provides this functionality and sections the information from the
other detectors off from the radio part to avoid confusion. Despite its name, it does not
hold any data from the other detectors itself, but offers access to ``HybridShower`` objects in
which this data is stored. For each additional detector (or set of detector data), a ``HybridSHower``
object can be added via the ``add_hybrid_shower`` method or accessed via the ``get_hybrid_shower``
or ``get_hybrid_showers`` methods.

It can be accessed via the ``get_hybrid_information'' method of the ``Event`` class.

Hybrid Shower
______________
The :class:`HybridShower<NuRadioReco.framework.hybrid_shower>` is
used to store information about a shower that was reconstructed with a complementary detector,
mainly via the parameter storage.

It can be accessed via the ``get_hybrid_shower`` and ``get_hybrid_showers`` methods of the
``HybridInformation`` class.

Hybrid Detector
_________________
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
