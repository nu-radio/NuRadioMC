import argparse
import numpy as np
import NuRadioReco.detector.RNO_G.rnog_detector
import NuRadioReco.modules.RNO_G.dataProviderRNOG
import NuRadioReco.modules.interferometricReconstruction
from NuRadioReco.examples.RNOG.processing import process_event
from NuRadioReco.utilities.framework_utilities import get_averaged_channel_parameter
from NuRadioReco.framework.parameters import (
    eventParameters as evp, channelParameters as chp, showerParameters as shp,
    particleParameters as pap, generatorAttributes as gta)
import datetime 

csw_info = {
  "PA" : [0,1,2,3],
  "PS" : [0,1,2,3,5,6,7],
  "ALL" : [0,1,2,3,5,6,7,9,10,22,23]
}
channels_to_include = csw_info["ALL"]

#helper function to load travel time maps
def load_maps_npz(path):
    ttcs = {}
    for ch in channels_to_include:
        path_ch = path + "/" + f"ttimes_{ch}.npz"
        ttcs[ch] = path_ch
    return ttcs

ttcs_npz = load_maps_npz("travel_time_maps") #replace with path to travel time maps 

parser = argparse.ArgumentParser(description='L2')
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--stat', type=int, required=True)
args = parser.parse_args()
filename = args.file
station_id = args.stat

detectorpath = "calibrated_detector.json" #path to calibrated detector file for station

det = NuRadioReco.detector.RNO_G.rnog_detector.Detector(detector_file = detectorpath)
det.update(datetime.datetime(2024, 3, 1))

reco = NuRadioReco.modules.interferometricReconstruction.InterferometricReco(det, station_id, ttcs_npz)
csw = NuRadioReco.modules.interferometricReconstruction.CSW(station_id, det)
scr = NuRadioReco.modules.interferometricReconstruction.SurfaceCorr(station_id, det)
dataProviderRNOG = NuRadioReco.modules.RNO_G.dataProviderRNOG.dataProvideRNOG()
dataProviderRNOG.begin(files = filename, det = det)

for idx, event in enumerate(dataProviderRNOG.run()):
    station = event.get_station(station_id) 
        
        
    r = 300 #change initial guess for r (spherical) based on event to be reconstructed 
    
    
    #change ranges and number of points according to event reconstucted and range of travel time maps used 
    #change path to where correlation map plots are saved as needed 


    #reconstruction returning just coordinate of maximum correlation and maximum correlation value 
    
    results = reco.run(event, station, channels_to_include, (-np.pi, np.pi), (-np.pi/2, np.pi/2), r, (-450, 150), (0,450), 250, 250, 180, 360, "plots")

    #reconstruction, coherently summed waveform (CSW) and surface correlation ratio (SCR) calculation 
    
    #returns correlation scores, time delays, travel time maps and correlation maps to be used for the CSW and SCR calculations 
    results = reco.run(event, station, channels_to_include, (-np.pi, np.pi), (-np.pi/2, np.pi/2), r, (-450, 150), (0,450), 250, 250, 180, 360, "plots", return_reco = True, return_score = True, return_delays = True, return_maps = True)
         
    #pass travel time maps, correlation scores, delay times and coordinate of max correlation point into here 
    csw_times, csw_values = csw.run(event, station, channels_to_include, results["maps"], results["maxcorr_coord"], results["score"], results["delays"])
        
    #pass correlation map and maximum correlation value into here 
    surf_corr_ratio, max_surf_corr, max_r, max_r = scr.run(results["reco"], results["maxcorr"])

dataProviderRNOG.end()
        
