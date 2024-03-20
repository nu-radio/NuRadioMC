The RNO-G detector class and database interface
===============================================

This page describes the RNO-G detector class and database which it is using. The description of the database will be limited to its data structure and reading interface, i.e., what is important in the context of the detector class. It is important to understand that the detector description of the RNO-G detector has to be time dependent for two reasons: I) The detector will change in time: New station will be deployed or maintenance at existing stations requires the description to change. II) Although the "true" detector description, which is unknown to us, is not changing our best description of it will change with time (e.g., a new calibration campaign will yield better estimations for the true position of the different antennas). These two reasons for the need of a time dependent detector description also bring us to the first concept to understand:


Detector time and database time
-------------------------------
To describe the detector at any time we use two different time variables, the `detector_time` and the `database_time`. The former is easy to understand, it defines the point in time at which we want to describe the detector (e.g., which station were deployed at that time? Which channels were commissioned and which electronic did those channel use?). The latter might not be immediately intuitive: It is used to define what measurment (or estimate) to use for each detector property. Imagine that you have only a poor estimation of the antenna positions initially due to the lack of a dedicated calibration campaign. However, for the time being this is your `primary` estimation (or measurement) for this property until you do better. Once you have done some calibration measurements and estimated more accurately the antenna positions, you will add them to the database as new primaries but keep the old values. The `database_time` will allow you to select the primary measurement, i.e., the best estimation of a certain property at the time you are running your reconstruction or simulation. Hence, by default the `database_time` will be set to `datatime.utcnow()`. That means also when you want to reanalyze older data (where at the time the detector was not know that well) you will select the best description of today. However, for special reason it might be interesting to reconstruct old or new data with the detector description known at an earlier point in time, the `database_time` allows you to do that by specifying it to some value in the past*.

Database data structure
-----------------------

The database is organized in collections. Each collection contains a list of objects which contain our data. The primary collection which describes our detector is the called `station_rnog`. This collection contains the `station list` which describes which station are deployed, when they were deployed. Each station has a list of its deployed channels. The information of the stations and channels positions is organized in seperate collections. Also the response of the different components in a channel's signal chain are stored in other collections. 



Detector class
--------------
