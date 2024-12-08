import NuRadioReco
import matplotlib.pyplot as plt
from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
import pandas as pd
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.modules import channelBandPassFilter
from NuRadioReco.detector import detector
import datetime
from NuRadioReco.modules import sphericalWaveFitter
from NuRadioReco.modules import channelAddCableDelay
import snr 
import rpr
import reco
import dedisperse_new
import csw 
import hilbert 
import impulsivity 

station_id = 12
detectorpath = "/data/i3store/users/avijai/RNO_season_2023.json"
channels_PA = [0,1,2,3]
channels_PS = [0,1,2,3,5,6,7]
channels_all = [0,1,2,3,5,6,7,9,10,22,23]
do_envelope = True
res = 100
solution = "direct_ice"

list_of_root_files = ['/data/i3store/users/avijai/station_runs/station12/run1611']

rpr = rpr.RPR()
csw = csw.CSW()
reco = reco.Reco()
snr = snr.SNR()
dedisperse = dedisperse_new.Dedisperse()
hilbert = hilbert.Hilbert()
impulsivity = impulsivity.Impulsivity()

readRNOGData = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData()
readRNOGData.begin(list_of_root_files, mattak_kwargs = {"backend":"pyroot"})

L2 = {}
for i_event, event in enumerate(readRNOGData.run()):
    L2[event.get_id()] = {}
    station_id = event.get_station_ids()[0]
    station = event.get_station(station_id)
    maxcorr_point, maxcorr = reco.run(event, station, detectorpath, station_id, channels_PA, do_envelope, res)
    avg_snr = snr.run(event, station)
    avg_rpr = rpr.run(event, station)
    #can change channels for CSW 
    csw_times, csw_values = csw.run(event, station, detectorpath, station_id, channels_PA, solution)
    csw_snr = snr.get_snr_single(csw_times, csw_values)
    csw_rpr = rpr.get_single_rpr(csw_times, csw_values)
    csw_hilbert_snr = hilbert.hilbert_snr(csw_values)
    csw_impulsivity = impulsivity.calculate_impulsivity_measures(csw_values)

    L2[event.get_id()]["max_correlation"] = maxcorr_point 
    L2[event.get_id()]["reco_vars"] = maxcorr 
    L2[event.get_id()]["avg_snr"] = avg_snr
    L2[event.get_id()]["avg_rpr"] = avg_rpr
    L2[event.get_id()]["csw_snr"] = csw_snr
    L2[event.get_id()]["csw_rpr"] = csw_rpr
    L2[event.get_id()]["csw_hilbert_snr"] = csw_hilbert_snr
    L2[event.get_id()]["csw_impulsivity"] = csw_impulsivity
    
    print(L2)

df = pd.DataFrame(L2)
df.to_csv("L2_12_1611.csv")



    
