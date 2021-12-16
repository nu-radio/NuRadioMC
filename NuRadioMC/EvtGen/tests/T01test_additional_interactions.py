import numpy as np
import h5py

from NuRadioMC.EvtGen import generator

datasets = {}
datasets['event_ids'] = np.arange(100, dtype=int)
datasets['zeniths'] = np.random.uniform(size=100)

additional_datasets = {}
additional_datasets['event_ids'] = np.array([0, 1, 10, 10, 10])
additional_datasets['zeniths'] = np.random.uniform(size=5)

generator.write_events_to_hdf5("T01.hdf5", data_sets=datasets, attributes={}, additional_interactions=additional_datasets)
generator.write_events_to_hdf5("T01_split10.hdf5", data_sets=datasets, attributes={}, additional_interactions=additional_datasets, n_events_per_file=10)

generator.split_hdf5_input_file("T01.hdf5", "T01_split10_later.hdf5", number_of_events_per_file=10)
