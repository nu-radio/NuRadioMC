from pymongo import MongoClient
import six
import os
import pymongo
import datetime
from astropy.time import Time
from pprint import pprint
import logging
import urllib.parse
import json
from bson import json_util #bson dicts are used by pymongo
import numpy as np
import pandas as pd
from NuRadioReco.utilities import units
import NuRadioReco.utilities.metaclasses
from NuRadioReco.detector import detector_mongo
from NuRadioReco.detector import detector

user="XXX"
pw="XXX"
mongo_detector = detector_mongo.Detector(database_connection=f"mongodb+srv://{user}:{pw}@cluster0-fc0my.mongodb.net/test?retryWrites=true&w=majority", database_name="RNOG_test")


"""
          tactical_name: string
              the name of the (long) tactical fiber being tested
          iglu_id: string
              the unique identifier of the iglu
          drab_id: string
              the unique name of the DRAB unit
          drab_channel: int
              the channel of the drab from which the measurement was taken
          breakout: int
              the connector from the fiber which
          temp: int
              the temperature at which the measurement was taken
          S_data: array of floats
              1st collumn: frequencies
              2nd/3rd collumn: S11 mag/phase
              4th/5th collumn: S12 mag/phase
              6th/7th collumn: S21 mag/phase
              8th/9th collumn: S22 mag/phase
          measurement_time: timestamp
              the time of the measurement
          primary_measurement: bool
              indicates the primary measurement to be used for analysis
          time_delay: array of floats
              the absolute time delay of each S parameter measurement (e.g. the group delay at
              a reference frequency)
          protocol: string
              details of the testing enviornment
"""

tactical_name="ref_channelXX"
iglu_id="ref_iglu"
drab_id="ref_drab"
breakout=1234
breakout_channel=1234
temp=20
S_data = np.zeros((9,100))
measurement_time=None
primary_measurement=Time.now().isot
time_delay= np.zeros(9)
protocol="Chicago_2022"
mongo_detector.downhole_chain(tactical_name, iglu_id, drab_id,
                         breakout, breakout_channel, temp, S_data, measurement_time,
                         primary_measurement, time_delay, protocol)
