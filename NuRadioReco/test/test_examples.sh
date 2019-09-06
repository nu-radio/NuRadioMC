set -e
cd NuRadioReco/examples/PhasedArray/Effective_volume
python T01generate_event_list.py minimal
python T02RunPhasedRNO.py --inputfilename minimal_eventlist.hdf5
python T02RunPhasedARA.py --inputfilename minimal_eventlist.hdf5
cd ../Noise_trigger_rate
python noise_trigger_rate.py --ntries 10
cd ../SNR_curves
python T01generate_event_list.py
python T02RunSNR.py 25.0deg_12/25.00_12_00_1.00e+18_1.26e+18/input/25.00_12_00_1.00e+18_1.26e+18.hdf5.part0001 proposalcompact_50m_1.5GHz.json config.yaml output_file.hdf5 output_snr.json
cd ../..
python FullReconstruction.py 32 example_data/example_data.hdf5 example_data/arianna_detector_db.json
