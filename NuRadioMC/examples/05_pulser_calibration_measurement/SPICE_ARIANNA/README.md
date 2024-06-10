# Pulser Calibration Measurement Example

This example shows how a pulser calibration measurement can be simulated in NuRadioMC. Here we simulate a SPICE core pulser drop observed by an ARIANNA shallow station. 

The A01*.py script generates the input event list, i.e., the pulser locations that should be simulated including all settings of the emitter. This example uses an artificial bandpasslimited delta pulse. 

The A02*.py script runs the NuRadioMC simulation. 

The A03*.py script reconstructs the signal arrival direction and the received electric field. 

The A04*.py script plots the results. 