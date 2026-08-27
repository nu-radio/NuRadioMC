# LOFAR Pipeline

This set of modules and scripts contain all that is used for the LOFAR simulation / data pipeline.

## Basic pipeline


### Data pipeline

`data_pipeline.py` : the base pipeline from raw LOFAR events -> reconstructed events. This included the full pipeline:

1. Read in the LOFAR data
2. Apply RFI filtering
3. Apply bandpass filters
4. Galactic calibration
5. Pulse finding
6. Plane wave direction fitting
7. Voltage to Electric Field conversion
8. Reconstruction with IFT model
9. Write event to .nur file

The intention is for steps 1 - 6 to be the standard pipeline, where the reconstruction (step 8) can be replaced with other reconstruction methods. 7 is optional, since not all methods use the V -> E converted results.

Steps 1 - 6 are wrapped in a module `NuRadioReco.modules.LOFAR.event_gen.dataEventGenerator`, which is called via `data_event_generator.process_event`.

All reconstruction methods (including the IFT model) is contained in `NuRadioReco.modules.LOFAR.reconstruction`. 

### Simulation pipeline

`simulation_pipeline.py` : the base pipeline from raw CoREAS events -> reconstructed events. This includes:

1. Read in the CoREAS star-shape simulations
2. Randomly vary core location & angles with LORA uncertainty
3. Interpolate to LOFAR antenna positions
4. Convert electric field to voltage
5. Resample trace to detector sampling rate
6. Bandpass using butterworth filter of order 10
7. Add galactic noise & Rayleigh noise with given skymodel & noise temperature
8. Voltage to Electric Field conversion
9. Reconstruction with IFT model
10. write event to .nur file

The intention is for steps 1 - 7 to be the standard pipeline, where the reconstruction (step 9) can be replaced with other reconstruction methods. 8 is optional, since not all methods use the V -> E converted results.

Steps 1 - 7 are wrapped in a module `NuRadioReco.modules.LOFAR.event_gen.CoREASEventGenerator`, which is called via `coreas_event_generator.process_event`.

All reconstruction methods (including the IFT model) is contained in `NuRadioReco.modules.LOFAR.reconstruction`. 

## TODOS

When including something, please make sure to document everything! This helps new users and old users alike to understand how it works and not include new (unneeded) functionality.

### General TODOS
- [x] include basic simulation pipeline
- [x] verify basic IFT reconstruction pipeline -> verified that it runs, but final reconstruction not checked
- [ ] include calculation of signal and electric field fluence to sim
- [ ] include requirements & documentation for using the pipeline -> added optional dependencies for niftyre for now, works with latest versions
- [ ] include CoREAS fluence reconstruction pipeline
- [ ] verify that round trip test works
- [ ] include validation tests -> should be place in `tests/LOFAR`
- [ ] simple CI/CD pipeline (soft, not enforced, but can be ran to verify installation works)
- [ ] better data structure to be better integrated with NuRadio format? (e.g. where to place the pipeline, now its in a not-so-good location)

### Visualisation TODOs
- [x] sample trace plots
- [x] polarization plots
- [ ] fluence plots
- [x] timing plots
- [x] electric field trace snapshots in CoREASEventGenerator

### Additional TODOs
- [x] move beamforming_utilities -> utilities
- [x] include LORA simulator for random direction & core as inputs (basic for now, based on average core / direction guess). Now the LORA simulator sets objects as a HybridShower
- [x] include LOFAR measured noise adder (takes .npy file, converts to .nur assuming same noise for all channels & stations)
- [ ] include / generalise to SKA -> have made the modules as independent as I can, so it should just be inheriting
- [ ] remove deprecated files
- [ ] include/improve with fixed interpolator version
- [ ] saving processed data as compressed hdf5 files?

## Some questions

- how to incorporate SKA into the framework? Basic functionality is the same for both LOFAR / SKA, except for the 
- what is already deprecated and can be removed?
- what code exists outside, and can be incorporated?
- naming convention for file for simulations -> should use the same event ID and store simulation run number with underscore? -> for now, using the simulation run number, and the directory structure shows event ID + mass, as with the 
- how to effectively integrate our code with NuRadio. Should there be an entire LOFAR pipeline repo? or a new directory with pipelines?


### Where everything is

`utilities/LOFAR` - utility functionalities for e.g. atmospheric models, IFT-related helpers, etc.
