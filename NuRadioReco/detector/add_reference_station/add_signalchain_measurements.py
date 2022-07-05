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
import datetime

user=os.getenv('RNOG_DB_USER')
pw=os.getenv('RNOG_DB_PW')
mongo_detector = detector_mongo.Detector(database_connection=f"mongodb+srv://{user}:{pw}@cluster0-fc0my.mongodb.net/test?retryWrites=true&w=majority", database_name="RNOG_test")

# Folder Path
path = "./box4_ps"
#Change directory
CWD=os.getcwd()
os.chdir(path)


# iterate through all files
for file in os.listdir():

    if file.endswith("MA.csv"):
        tactical_name="ref_channel"+file.split('_')[-2].strip('CH')
        iglu_id="ref_iglu_"+file.split('_')[-3]
        drab_id="ref_drab_DAQ4"
        file_path = f"{path}/{file}"
        df = pd.read_csv(file_path,
                                  skiprows=6,
                                  skipfooter=2,
                                  usecols=list(range(0,9)))
        df['Freq(Hz)']*=units.Hz
        S_data = np.array(df).T

        file_gd = file.strip("MA.csv")+"GroupDelay.csv"
        file_path_gd = f"{path}/{file_gd}"
        df_gd = pd.read_csv(file_path_gd,
                                  skiprows=6,
                                  skipfooter=2)
        time_delay_cut = df_gd[(df_gd['Freq(Hz)']>200*10**6) & (df_gd['Freq(Hz)']<600*10**6)]
        time_delay =  np.full(9, (time_delay_cut['S21 Delay(s)'].mean())*units.s)

        breakout=1234
        breakout_channel=1234
        measurement_time=datetime.datetime.utcnow()
        temp = 20
        primary_measurement=True
        protocol="Chicago_2022"
        mongo_detector.downhole_chain(tactical_name,
                                  iglu_id,
                                  drab_id,
                                  breakout,
                                  breakout_channel,
                                  temp,
                                  S_data,
                                  measurement_time,
                                  primary_measurement,
                                  time_delay,
                                  protocol)

os.chdir(CWD)

# Folder Path
surf_path = "./Surface_cables_box2"
#Change directory
os.chdir(surf_path)

for file in os.listdir():
    if file.endswith("_wcables_-60_chan_1_MA.csv"):
        tactical_name="ref_surface"
        surface_id = "Amp_Surface_"+file.split('_')[-5]
        surface_channel = "0"
        file_path = f"{surf_path}/{file}"
        df = pd.read_csv(file_path,
                        skiprows=6,
                        skipfooter=2,
                        usecols=list(range(0,9)))
        df['Freq(Hz)']*=units.Hz
        S_data = np.array(df).T
        
        file_gd = file.strip("MA.csv")+"GroupDelay.csv"
        file_path_gd = f"{surf_path}/{file_gd}"
        df_gd = pd.read_csv(file_path_gd,
                                  skiprows=6,
                                  skipfooter=2)
        time_delay_cut = df_gd[(df_gd['Freq(Hz)']>200*10**6) & (df_gd['Freq(Hz)']<600*10**6)]
        time_delay =  np.full(9, (time_delay_cut['S21 Delay(s)'].mean())*units.s)
        
        measurement_time=datetime.datetime.utcnow()
        temp = 20
        primary_measurement=True
        protocol="Chicago_2022"
        mongo_detector.surface_chain(tactical_name,
                                      surface_id,
                                      surface_channel,
                                      temp,
                                      S_data,
                                      measurement_time,
                                      primary_measurement,
                                      time_delay,
                                      protocol)
