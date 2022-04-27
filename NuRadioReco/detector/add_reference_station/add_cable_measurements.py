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
from glob import glob
import sys
import pandas as pd

user=os.getenv('RNOG_DB_USER')
pw=os.getenv('RNOG_DB_PW')
mongo_detector = detector_mongo.Detector(database_connection=f"mongodb+srv://{user}:{pw}@cluster0-fc0my.mongodb.net/test?retryWrites=true&w=majority", database_name="RNOG_test")

# Folder Path
path = sys.argv[1]
#Change directory
os.chdir(path)

files = glob("*_LM.csv")

# iterate through all files
for LMfile in files:
    if "BASELIN" in LMfile:
        continue
    if LMfile.endswith("_LM.csv"):
        print(LMfile)
        df = pd.read_csv(LMfile,
                         skiprows=17,
                         skipfooter=1,
                         engine="python")
        LM_data = np.array(df.values).T

        Pfile = LMfile.replace("_LM.csv", "_P.csv")
        if not os.path.exists(Pfile):
            print("pfile does not exist...", Pfile)
            continue
        try:
            df = pd.read_csv(Pfile,
                         skiprows=17,
                         skipfooter=1,
                         engine="python")
        except:
            continue
        P_data = np.array(df.values).T

        cable_name = Pfile.split("/")[-1].replace("_P.csv","")
        primary_measurement = True
        protocol = "Chicago_2020"
        print("inserting:", cable_name, np.shape(LM_data), np.shape(P_data), primary_measurement, protocol)
        mongo_detector.CABLE_add_Sparameters(cable_name, LM_data, P_data, primary_measurement, protocol)
        #break
