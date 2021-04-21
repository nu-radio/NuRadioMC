import sys
import numpy as np
from NuRadioReco.detector import detector
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.modules.neutrinoDirectionReconstructor import voltageToEfieldAnalyticConverterForNeutrinos
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units, fft, trace_utilities
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import eventParameters as ep
import h5py


vTEACFN = voltageToEfieldAnalyticConverterForNeutrinos.voltageToAnalyticEfieldConverterNeutrinos()

det = detector.Detector(json_filename = "./configurations/ARIANNA_4LPDA_1dipole.json", antenna_by_depth=False)
icemodel = medium.get_ice_model("southpole_2015")

passband_low = {}
passband_high = {}
for channel_id in range(0, 5):
    passband_low[channel_id] = [1 * units.MHz, 500 * units.MHz]
    passband_high[channel_id] = [80 * units.MHz, 1000 * units.GHz]

def getWeights(file,event):
    with h5py.File(file, 'r') as f:
        maskId = f['event_group_ids'][:] == event
        return(f['weights'][maskId][0])

def getVrms(file,station_id):
    with h5py.File(file, 'r') as f:
        return f["station_{:d}".format(station_id)]['Vrms']


#fout["station_{:d}".format(station_id)].attrs['Vrms'] = list(self._Vrms_per_channel[station_id].values())

def forward_folding(event):
    dataNurFile = "./data/triggeredNeutrinoEvents.nur"
    dataHDF5File = "./data/triggeredNeutrinoEvents.hdf5"
    savFile = "./data/reconstructedNeutrinoProperties"
    use_channels = [0,1,2,3,4]
    template = NuRadioRecoio.NuRadioRecoio(dataNurFile)
    evt = template.get_event_i(event)
    for station in evt.get_stations():
        weight = getWeights(dataHDF5File,evt.get_run_number())
        triggers = station.get_triggers()

        sim_shower = evt.get_first_sim_shower()
        nu_zenith_sim = sim_shower[shp.zenith]
        nu_azimuth_sim = sim_shower[shp.azimuth]
        shower_energy_sim = sim_shower[shp.energy]

        sim_station = station.get_sim_station()
        efields = sim_station.get_electric_fields()
        station.set_parameter(stnp.zenith,efields[0].get_parameter(efp.zenith))
        station.set_parameter(stnp.azimuth,efields[0].get_parameter(efp.azimuth))

        det.update(station.get_station_time())
        station_id = station.get_id()

        # Check this number, this is the noise RMS for an 80-500MHz bandwidth:
        noise_RMS = 8.521169476538645e-06
        # Other methods for this
        # getVrms(dataHDF5File, station_id)   # should get Vrms values from hdf5 iff they were origionally stored
        # det.get_noise_RMS(station_id, 0)    # Can get from detector object but software will need to be updated, at the time of writing this, this attribute was not correct

        # Caculates the noiseless SNR (i.e. max signal amplitude without noise divided by noise Vrms)
        tmp = {}
        for sim_channel in sim_station.iter_channels():
            channel_id = sim_channel.get_id()
            signal_amplitude = sim_channel[chp.maximum_amplitude_envelope]
            if channel_id not in tmp:
                tmp[channel_id] = []
            tmp[channel_id].append(signal_amplitude)

        SNRs = []
        for channel_id in use_channels:
            trace = station.get_channel(channel_id).get_trace()
            if channel_id in tmp:
                SNRs.append(max(tmp[channel_id])/noise_RMS)
            else: # Sometimes this channel may be in the shadow zone and not have an electric field solution. Prevents division by 0
                SNRs.append(-1)

        # This is the line that actually does all of the neutrino reconstruction and stores some of the station parameters
        vTEACFN.run(evt, station, det, icemodel, use_channels=use_channels, debug=False, use_bandpass_filter=True, passband_low=passband_low, passband_high=passband_high, hilbert=False, attenuation_model='SP1', shower_type='HAD', parametrization='Alvarez2009')

        nu_zenith_reco = station.get_parameter(stnp.nu_zenith)
        nu_azimuth_reco = station.get_parameter(stnp.nu_azimuth)
        shower_energy_reco = station.get_parameter(stnp.shower_energy)

        # Create output data structure for another plotting script to use
        output = {}
        output["SNRs"] = SNRs
        output["weight"] = weight
        output["triggers"] = triggers
        output["nu_zenith_sim"] = nu_zenith_sim
        output["nu_zenith_reco"] = nu_zenith_reco
        output["nu_azimuth_sim"] = nu_azimuth_sim
        output["nu_azimuth_reco"] = nu_azimuth_reco
        output["shower_energy_sim"] = shower_energy_sim
        output["shower_energy_reco"] = shower_energy_reco

        # Saves events indidivually. This is good for submitting this script as a bunch of parallel jobs per event since the nu reco takes roughly 30 minutes per events to run.
        # Then concatenating the resulting files together to make one final data file for analysis
        np.save(savFile + '_' + str(event),output)


def main():
    # Argument is the event number being used for analaysis.
    if len(sys.argv) == 2:
        forward_folding(int(sys.argv[1]))
    else:
        eventNum = 0
        forward_folding(eventNum)

if __name__== "__main__":
    main()
