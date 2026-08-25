# LOFAR Pipeline

This set of modules and scripts contain all that is used for the LOFAR simulation / data pipeline.

## Basic pipeline



## TODOS

When including something, please make sure to document everything! This helps new users and old users alike to understand how it works and not include new (unneeded) functionality.

### General TODOS
- [x] include basic simulation pipeline
- [ ] verify basic IFT reconstruction pipeline
- [ ] include requirements & documentation for using the pipeline
- [ ] include CoREAS fluence reconstruction pipeline
- [ ] verify that round trip test works
- [ ] include validation tests
- [ ] simple CI/CD pipeline (soft, not enforced, but can be ran to verify installation works)
- [ ] better data structure to be better integrated with NuRadio format? (e.g. where to place the pipeline, now its in a not-so-good location)

### Visualisation TODOs
- [x] sample trace plots
- [x] polarization plots
- [ ] fluence plots
- [x] timing plots

### Additional TODOs
- [x] move beamforming_utilities -> utilities
- [x] include LORA simulator for random direction & core as inputs (basic for now, based on average core / direction guess)
- [ ] include LOFAR measured noise adder
- [ ] include / generalise to SKA
- [ ] remove deprecated files
- [ ] include/improve with fixed interpolator version
- [ ] saving processed data as compressed hdf5 files?

## Some questions

- how to incorporate SKA into the framework? Basic functionality is the same for both LOFAR / SKA, except for the 
- what is already deprecated and can be removed?
- what code exists outside, and can be incorporated?
- naming convention for file for simulations -> should use the same event ID and store simulation run number with underscore?
- how to effectively integrate our code with NuRadio. Should there be an entire LOFAR pipeline repo? or a new directory with pipelines?


### Contents

`utilities/LOFAR` - utility functionalities for e.g. atmospheric models, IFT-related helpers, etc.
