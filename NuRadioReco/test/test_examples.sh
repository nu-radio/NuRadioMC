set -e
# Phased array
cd NuRadioReco/examples/PhasedArray/Effective_volume
python3 T01generate_event_list.py minimal
python3 T02RunPhasedRNO.py --inputfilename minimal_eventlist.hdf5
# python3 T02RunPhasedARA.py --inputfilename minimal_eventlist.hdf5 # Deprecated, for the time being
rm minimal_eventlist.hdf5
rm output_PA*hdf5
cd ../Noise_trigger_rate
python3 T01MeasureNoiselevel.py --ntries 10
cd ../SNR_curves
python3 T01generate_event_list.py
python3 T02RunSNR.py --inputfilename 25.0deg_12/25.00_12_00_1.00e+18_1.26e+18/input/25.00_12_00_1.00e+18_1.26e+18.hdf5 --detectordescription ../Effective_volume/8antennas_100m_0.5GHz.json --config config.yaml --outputfilename output_file.hdf5 --outputSNR output_snr.json
rm -rf *deg_12/
rm output_file.hdf5
rm output_snr.json
# Envelope phased array
#cd ../../EnvelopePhasedArray/Effective_volume/
#python3 T01generate_event_list.py minimal
#python3 T02RunPhasedRNO_shortband.py --inputfilename minimal_eventlist.hdf5
#rm minimal_eventlist.hdf5
#rm output*hdf5
#cd ../Noise_trigger_rate
#python3 noise_trigger_rate.py --ntries 10
## Alias phased array
#cd ../../AliasPhasedArray/SNR_study
#python3 T01_generate_events_simple.py --n_events 10
#python T02SNRNyquist.py input_alias_SNR.hdf5 phased_array_100m_0.25GHz.json config.yaml out.hdf5 out.json --nyquist_zone 2 --upsampling_factor 4 --noise_rate 1
#rm input_alias_SNR.hdf5
#rm out.hdf5
#rm out.json
#cd ../Noise_trigger_rate
#python3 noise_trigger_rate.py --ntries 10
# Full reconstruction
cd ../..
python3 FullReconstruction.py 32 example_data/example_data.hdf5 example_data/arianna_station_32.json
python3 read_full_CoREAS_shower.py example_data/example_data.hdf5
python3 SimpleMCReconstruction.py
python3 CustomHybridDetector.py
