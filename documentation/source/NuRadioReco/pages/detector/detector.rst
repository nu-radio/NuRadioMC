Detector Description
=========================================
NuRadioReco provides detector description classes that allow access to any
relevant information about the state of the radio detector at a given time.
There are two classes that provide this functionality: The
:class:`Detector <NuRadioReco.detector.detector.Detector>` and the
:class:`GenericDetector <NuRadioReco.detector.generic_detector.GenericDetector>`.
Both are used very similarly, with the ``GenericDetector`` being more for
simulation studies.

For the RNO-G experiment, there is an additional detector class
:class:`rnog_detector <NuRadioReco.detector.RNO_G.rnog_detector>`.
Some more details about this detector class and its implementation are given
:doc:`here </NuRadioReco/pages/detector/rnog>`.


The Detector Class
-----------------------------
The ``Detector`` class provides easy access to information about a given detector.
Instead of manually sending requests to an SQL database or parsing through
JSON files, a ``Detector`` object is created and given the appropriate SQL
configuration or JSON detector description and information about the detector
is requested via *get*-methods. By setting the ``Detector`` to a specific time
via the ``update`` function, it provides the detector configuration as it was
at  specific time.

.. code-block:: python

  import NuRadioReco.detector.detector
  import astropy.time

  det = NuRadioReco.detector.detector.Detector(source='json', json_filename='path/to/json/file')
  detector_time = astropy.time.Time('2018-01-01 20:00:00')
  det.update(detector_time) # set the detector to the state it was in at detector_time

  station_position = det.get_absolute_position(42)

.. Important:: Since the detector configuration can change with time, the detector has to be set to the correct time using the ``update`` method.


Detector Description Formats
-----------------------------
The detector description consists of several tables for the different component
types that can reference each other by their IDs. For example, each channel
in the channels table has a Channel ID and a Station ID that associates it
with a station from the stations table.

The channel table contains references to the individual components a channel consists of such as amplifier, antenna, cables, etc.
The components are saved in separate tables.

Since station configurations can change over the lifetime of a detector, each station and channel entry
has a commission and decommission time which specifies when it was part of the
detector in the given configuration. If a configuration is changed, the decommission
time is set to the time when the change took place and a new channel (or complete station) is added
with the new configuration. The ``Detector`` will then use these entries to
automatically return the correct detector configuration for a given point in time.
The components itself do not carry a comission/decomission time. This reflects the procedures of a large experiment,
every component is calibrated and its information is added into the database. Later on, each channel is constructed out
of the different components, and finally a station is composted out of multiple channels.



DataBase
_______________

Information about the detector can be stored in a remote MySQL database, which has
the advantage that one always uses the latest detector information, as long as the
database is properly maintained. Since MySQL requests can be rather slow, entries
are buffered once they have been requested from the MySQL server.

JSON
_______________
In some situations, e.g. when doing a simulation study, it is useful to be able
to use a variety of detector descriptions without the need to set up a MySQL database
every time. Therefore, the NuRadioReco ``Detector`` and ``GenericDetector`` class
can also read in a JSON file, in which information about the detector is stored
as key-value pairs. JSON files for both the ARIANNA and ARA detector are shipped
with NuRadioReco and can be modified to simulate different detector configurations.

We also provide a converter that dumps the SQL database into a JSON file that then can be used offline.


Dictionary
_______________
In principle, the detector description can also be passed to the ``Detector``
or ``GenericDetector`` as a python dictionary. This is used internally by NuRadioReco,
e.g. when loading the detector description from an event file. For most applications,
we recommend not using this feature and storing the description in a JSON file instead.

Signal Chain Responses
_______________________

GenericDetector
----------------------------

For simulation studies, it is sometimes necessary to simulate a detector
consisting of a large number of mostly identical stations. In another situation,
a station may have a large number of channels that only differ by a few characteristics,
e.g. the antenna position. In these cases, creating a full table entry for each
station or channel would be cumbersome, error-prone and make it harder to make
changes later.
For this reason, *NuRadioReco* provides the ``GenericDetector`` detector class.
The ``GenericDetector`` allows to declare a *reference station*, which has to be fully
described, but for all other stations the description can be incomplete.
When a station is requested, the  ``GenericDetector`` substitutes any
missing entries with the entries from the *reference station*. If the channels for
a station are requested and none are found, it is assumed that it has the same
channels as the *reference station* and the channels of the *reference station* are
returned.
In the same way, it is also possible to specify *reference channels*. When a requested
entry is missing
in one of the other channels, it is substituted with the entry in the *reference channel*.

*Reference stations* and *reference channels* are declared for each channel in the detector
description JSON file. A simple detector description using this feature might look like this:


.. code-block:: json

  {
    "channels": {
      "0": {
        "ant_comment": "phased array",
        "amp_type": "iglu",
        "adc_sampling_frequency": 2.4,
        "adc_n_samples": 1024,
        "ant_orientation_phi": 0.0,
        "ant_orientation_theta": 0.0,
        "ant_position_x": 0.0,
        "ant_position_y": 0.0,
        "ant_position_z": -100.0,
        "ant_rotation_phi": 90.0,
        "ant_rotation_theta": 90.0,
        "ant_type": "RNOG_vpol_4inch_center_n1.73",
        "cab_time_delay": 0.0,
        "channel_id": 0,
        "station_id": 1
      },
      "1": {
        "ant_position_z": -200,
        "channel_id": 1,
        "station_id": 1,
        "reference_channel": 0
      }
    },
    "stations": {
      "1" : {
        "pos_altitude": 0.0,
        "pos_easting": 0.0,
        "pos_northing": 0.0,
        "pos_site": "summit",
        "station_id": 1,
        "station_type": null
      },
      "2" : {
        "pos_easting": 1000,
        "station_id": 2,
        "reference_station": 1
      }
    }
  }

In this case, only channel 0 and station 1 are fully defined. But if we queried for channel 1,
any missing property will be filled in with the corresponding property from channel 0. Likewise,
station 2 will take any missing entries from station 1 and, since no channels were explicitly
defined for station 2, queries for station 2's channels will return copies of station 1's channels.


.. Important:: In older versions of *NuRadioMC*, reference stations and channels were defined by parameters
  passed to the ``GenericDetector``'s ``__init__`` method. While this still works for compatibility,
  it is deprecated and should not be used.


When creating a new ``Detector`` object, the constructor will check if the detector description
contains any ``reference_station`` or ``reference_channel`` entries, and create a GenericDetector
if these entries are found.


.. Important:: The ``GenericDetector`` does not support commission and decommission times.
  It can therefore not give a time-dependent detector description and should only be used
  for simulation studies, never to reconstruct real data.

Event-Specific Changes
______________________
In some situations, events in the same file can have a different detector description.
While these situations should be avoided, doing so would sometimes be too cumbersome,
so the ``GenericDetector`` offers a way to store event-specific changes to the
detector.
In addition to the normal detector description, the method ``add_station_properties_for_event``
allows it to be given a list of properties that are different for a given event
and station. The ``GenericDetector`` can be set to a specific event via the
``set_event`` method and will return the detector configuration for that event
if queried afterwards.

The process thereby is as follows: First the data from the detector description
is read. Then any missing entries are substituted by those from the *default*
station. Finally, if any event-specific changes for the current station and event are
registered, the properties in question are replaced and the station is returned.

One usage example are star-pattern CoREAS air shower simulations where every simulation has different station positions.
Here, only the station positions are different between each event and saved at _event specific changes to the detector
description.

Detector Description in Event Files
-----------------------------------
To make it easier to keep track of which detector description was used in the reconstruction
of a given event file, it is possible to store the detector description in an
event file and read it along with the events.

Writing the Detector
____________________
To write a detector description into an event file, the detector description
is passed to the ``run`` method of the ``EventWriter`` module. In order to
keep the file size small, only information about channels and stations that
are used in the saved events are written into the event file.

.. code-block:: python

  import NuRadioReco.modules.io.eventReader
  import NuradioReco.modules.io.eventWriter
  import NuRadioReco.detector.detector
  import astropy.time

  det = NuRadioReco.detector.detector.Detector(source='json', json_filename='path/to/json/file')
  detector_time = astropy.time.Time('2018-01-01 20:00:00')
  det.update(detector_time)

  event_reader = NuRadioReco.modules.io.eventReader.EventReader()
  event_reader.begin(['path/to/file'])

  event_writer = NuRadioReco.modules.io.eventWriter()
  event_writer.begin('output_filename.nur')
  for event in event_reader.run():
    event_writer.run(event, det=det)

Reading the Detector
____________________
To access the detector description in an event file, the ``EventReader`` and
``NuRadioRecoio`` modules provide the ``get_detector`` method, which always
returns the detector for the last file from which an event was requested. If
the detector in the file is a ``GenericDetector``, its ``set_event`` method
will also be called automatically in case there are event-specific changes to
the detector.

In order to use this feature, the parameters ``parse_detector`` and ``read_detector``
have to be set to ``True`` for  constructors of the ``NuRadioRecoio`` and
``EventReader`` modules, respectively.

.. code-block:: python

  import NuRadioReco.modules.io.eventReader
  event_reader = NuRadioReco.modules.io.eventReader.EventReader()
  event_reader.begin(['path/to/file'], read_detector=True)
  for event in event_reader.run():
    det = event_reader.get_detector()

.. Important:: When reading multiple files with different detector descriptions, ``get_detector`` needs to be called
 each time an event from another file is read to get the correct ``Detector`` or ``GenericDetector``.

 We recommend calling ``get_detector`` after every new event request.

Detector Viewer
_________________
NuRadioReco provides a visualization for detector description. It is stored in the folder
``NuRadioReco/detector/detector_browser`` and works similar to the :doc:`event viewer</NuRadioReco/pages/event_display>`.

To start it, execute the command ``python index.py /path/to/folder``. Then open
a web browser and go to the URL printed on the terminal (http://127.0.0.1:8080/
by default).

The browser can read both the standard Detector and the GenericDetector format,
as well as detector descriptions stored in event files. The input format can
be selected from the dropdown, after which the detector file can be opened.
