### Use following commands to run this example:

python A01generate_pulser_events.py

python runARA02.py emitter_event_list.hdf5 ARA02Alt.json config.yaml output.hdf5 output.nur

### Inside A01generate_pulser_events.py ###

Define 'simulation_mode' as "emitter" to work with radio emitting pulser models

For "emitter_antenna_type" , all available antenna models in NuRadioMC can be used

The amplitude of source voltage signal ( waveform ) including magnitude and dimesion (unit) is defined through "emitter_amplitude"

For tone_burst and square model the half width of the signal can be defined via; "half_of_pulse_width"

And for cw and tone_burst models the signal frequency is defined by "emitter_frequency"

### ARA02Alt.json ###


Instead of usual VPol and HPol antennas (bicone and fourslot),  ARA02Alt.json uses "XFDTD_Vpol_CrossFeed_150mmHole_n1.78" and "XFDTD_Hpol_150mmHole_n1.78"

( these antennas have relatively low cross polarization)

### hdf5 files ###


The root data containing waveforms generated in University of Kansas (KU) lab for IDL and HVSP2 signals, have been converted in to hdf5 files (ex: idl_data.hdf5, hvsp2_data.hdf5).

These files contain voltage array ( normalized to one) volt and corresponding time array converted in nanoseconds.

## "emitter_model" ####
There are many available radio emitting models including cw, square , tone_burst, idl, hvsp2 etc.. (for more detail, visit :  NuRadioMC/SignalGen/emitter.py )

