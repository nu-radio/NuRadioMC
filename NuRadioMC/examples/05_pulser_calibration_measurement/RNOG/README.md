### RNO-G Calibration Pulser Simulation

Simulates RNO-G calibration pulser events using measured lab waveform templates
(`rno_cal5C_*dB`) with the full RNO-G hardware response chain (amplifiers,
filters, cable delays) and realistic deep high-low trigger.

### Quick start

    # Generate events for a single position
    python A01generate_pulser_events.py --station 23 --az 45 --zen 110 --r 200

    # Run the simulation
    python A02RunSimulation.py data/events_r200.0_zen110.0_az45.0.hdf5 config.yaml output.hdf5 output.nur

Each axis (`--az`, `--zen`, `--r`) accepts either a single fixed value or a
range as `start stop step`. A single value fixes that coordinate while the
others vary:

    # Scan azimuth and zenith at a fixed distance of 35 m
    python A01generate_pulser_events.py \
        --station 23 \
        --az 0 360 30 --zen 80 160 10 --r 35 \
        --n-events 50

    # Full 3D grid
    python A01generate_pulser_events.py \
        --station 23 \
        --az 0 360 30 --zen 80 160 10 --r 50 300 50 \
        --n-events 50

This creates named event files in `data/` (e.g. `events_r50.0_zen80.0_az0.0.hdf5`).
Then run A02 on each grid point, e.g. via SLURM array jobs
(see `submit_A02.sh`).

### Emitter models

Uses measured RNO-G calibration pulser waveforms from Chicago lab (2025).
Available attenuation settings: 0, 5, 10, 15, 20 dB.
The 0 dB pulses are saturated in the iglu amplifier.
Templates are at `NuRadioMC/data/RNO_G_pulser_waveforms/`.

### Emitter position

Position is specified in spherical coordinates (azimuth, zenith, distance)
relative to the station's phased array center (midpoint of channels 1 and 2),
with the z-axis pointing upward (zenith = 0 is directly above, zenith = 180
is directly below). Antenna positions are read from the RNO-G MongoDB
detector description.

### Hardware response and trigger

The simulation applies the full RNO-G signal chain via
`NuRadioReco.modules.RNO_G.hardwareResponseIncorporator`, then evaluates
a realistic deep high-low trigger: FLOWER board digitization
(`triggerBoardResponse`) followed by a 2-fold coincidence high-low threshold
on the phased array channels (0-3) at a 1 Hz singles rate.
