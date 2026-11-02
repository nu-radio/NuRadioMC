# LOFAR Pipeline

This set of modules and scripts contain all that is used for the LOFAR simulation / data pipeline.

## Directory structure

- `event_gen/` : consists of modules that generate processed events (from data, coreas simulations, etc)
- `reconstruction/` : consists of modules that uses processed events for reconstruction.

Otherwise all modules placed at the parent level are related to the base pipeline build (aside from other modules used that are appropriately placed elsewhere in the repo.)

NOTE: all modules in subdirectories have import functions contained in the parent-level `init.py` so that module imports can be used as before.

## TODOS

When including something, please make sure to document everything! This helps new users and old users alike to understand how it works and not include new (unneeded) functionality.

- [ ] fix the beamforming code (borrow functionality from Subhadip's version?)
- [ ] include coreas fluence reconstruction module
- [ ] include smiet fluence reconstruction module
- [ ] remodel the structure such that we have air_shower_detection -> reconstruction, event_generation, process -> LOFAR / SKA if there is specific ones for that


## Some questions

- how to incorporate SKA into the framework? Basic functionality is the same for both LOFAR / SKA, except for the 
- what is already deprecated and can be removed?
- what code exists outside, and can be incorporated?
- naming convention for file for simulations -> should use the same event ID and store simulation run number with underscore?
- how to effectively integrate our code with NuRadio. Should there be an entire LOFAR pipeline repo? or a new directory with pipelines?
