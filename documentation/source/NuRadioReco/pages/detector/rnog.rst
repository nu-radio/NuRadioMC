The RNO-G detector class and database interface
===============================================

This page describes the RNO-G detector class and database which it is using. The description of the database will be limited to its data structure and reading interface, i.e., what is important in the context of the detector class. It is important to understand that the detector description of the RNO-G detector has to be time dependent for two reasons: I) The detector will change in time: New station will be deployed or maintenance at existing stations requires the description to change. II) Although the "true" detector description, which is unknown to us, is not changing our best description of it will change with time (e.g., a new calibration campaign will yield better estimations for the true position of the different antennas). These two reasons for the need of a time dependent detector description also bring us to the first concept to understand:


Detector time and database time
-------------------------------
To describe the detector at any time we use two different time variables, the `detector_time` and the `database_time`. The former is easy to understand, it defines the point in time at which we want to describe the detector (e.g., which station were deployed at that time? Which channels were commissioned and which electronic did those channel use?). The latter might not be immediately intuitive: It is used to define what measurment (or estimate) to use for each detector property. Imagine that you have only a poor estimation of the antenna positions initially due to the lack of a dedicated calibration campaign. However, for the time being this is your `primary` estimation (or measurement) for this property until you do better. Once you have done some calibration measurements and estimated more accurately the antenna positions, you will add them to the database as new primaries but keep the old values. The `database_time` will allow you to select the primary measurement, i.e., the best estimation of a certain property at the time you are running your reconstruction or simulation. Hence, by default the `database_time` will be set to `datatime.utcnow()`. That means also when you want to reanalyze older data (where at the time the detector was not know that well) you will select the best description of today. However, for special reason it might be interesting to reconstruct old or new data with the detector description known at an earlier point in time, the `database_time` allows you to do that by specifying it to some value in the past*.

Database structure
------------------

The database is organized in collections. Each collection contains a list of objects which contain our data. The primary collection which contains the `station list`, which lists the deployed stations and their channels, is the called `station_rnog`. The information of the stations and channels positions is organized in separate collections. The response of the different components in a channel's signal chain are stored in other collections. Each component class (i.e. coax cable, fiber, DRAB, IGLU, ...) has its own collection. A schematic of the database structure is shown below.

.. image:: rnog-mongo-database-structure.pdf
  :width: 100%

Signal Chain
------------

Each channel has a "signal chain" which is basically a list of all the individual reponses which are necessary to describe the entire analog response of this channel. This list is implemented as dictionary, the key of the dictionary is also the name of the collection in which it looks for the specified response (value). The key can have a suffix like `_x`, with `x` being an integer, which allows to specify several responses from the same collection to be added to the signal chain.


Response class
--------------

For the response of a channel the detector returns object of the Response class. This class has implemented operators to apply the response to trace objects. These objects store the time delay (= the group delay at ~ 200MHz) and have removed this time delay from the group delay (= the S21 parameter). The group delay is calculated by the hardware database as read from there. Keep in mind that S21 parameter in the data typically have the full time delay. Only when this data is accesses thought the detector / response class this time delay is removed from the response function.

Detector class
--------------

Coming soon (Hopefully).
