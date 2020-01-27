set -e
cd NuRadioReco/examples/PhasedArray/Effective_volume
python3 T01generate_event_list.py minimal
python3 T02RunPhasedRNO.py --inputfilename minimal_eventlist.hdf5
python3 T02RunPhasedARA.py --inputfilename minimal_eventlist.hdf5
cd ../Noise_trigger_rate
python3 noise_trigger_rate.py --ntries 10
cd ../SNR_curves
python3 T01generate_event_list.py
python3 T02RunSNR.py 25.0deg_12/25.00_12_00_1.00e+18_1.26e+18/input/25.00_12_00_1.00e+18_1.26e+18.hdf5.part0001 proposalcompact_50m_1.5GHz.json config.yaml output_file.hdf5 output_snr.json
cd ../../EnvelopePhasedArray/Effective_volume/
python3 T01generate_event_list.py minimal
python3 T02RunPhasedRNO_shortband.py --inputfilename minimal_eventlist.hdf5
cd ../Noise_trigger_rate
python3 noise_trigger_rate.py --ntries 10
cd ../..
python3 FullReconstruction.py 32 example_data/example_data.hdf5 example_data/arianna_station_32.json
python3 read_full_CoREAS_shower.py example_data/example_data.hdf5
python3 CustomHybridDetector.py
