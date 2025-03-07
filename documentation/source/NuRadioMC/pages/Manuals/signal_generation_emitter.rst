Signal Generation (emitter)
====================================
For cases where the radio signals don't come from a neutrino interaction but from a pulser, the emitter module of NuRadioMC can be used for the simulation.  To use these pulses the input file must contain the key 'emitter_model' which specifies what shape the emitter pulse should have. There are several simple pulse options available such as 'delta_pulse', 'cw', 'square', and 'gaussian'. More complex models are explained in the following sections.

SPice Pulser
--------------
The pulses used by the module come from an anechoic chamber measurement. To use these pulses the 'emitter_model' must be set to 'efield_idl1_spice'. Ten pulses were recorded for each of these specific launch angles (0deg, 15deg. 30deg, 45deg, 60deg, 75deg, 90deg). To account for the difference in dielectric environments, after reconstructing the anechoic chamber electric field, the frequency content is shifted from an in-air medium to an in-ice medium by dividing by the index of refraction of deep ice (n = 1.78). Shifting the frequencies by 1/n serves as a first-order approximation since the antenna is wavelength-resonant. After performing this frequency correction, a rectangular band pass filter between 80 MHz to 300 MHz is applied in order to remove unwanted noise. See `(Anker et al., 2020) <https://iopscience.iop.org/article/10.1088/1748-0221/15/09/P09039>`__ for more details. Additionally, the amplitude of the pulses was scaled to a propagation distance of 1 m.

