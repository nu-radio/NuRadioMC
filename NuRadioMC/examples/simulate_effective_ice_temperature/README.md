# Simulation of the effective temperature at an antenna
This code is originally from Steffen and adapted for NuRadio from Felix.

This code uses RadioPropa to propagate rays from an antenna into the ice. The rays are reflected at the surface (emission from above the surface is ignored). Power lost at the reflection point is taken into account.
The effective temperature of the ice for a given direction include the temperature and attenuation profile of the ice along the ray paths.


## Requirements
* git@github.com:RNO-G/antenna-positioning.git
  * At the time of writing this example requires a specific branch of RadioPropa: ice_model/exponential_polynomial
* git@github.com:nu-radio/RadioPropa.git


Notes:
Be aware that the calculated temperature is the external noise temperature AT THE ANTENNA. I.e. one would need to treat this temperature similar to the Galactic noise adder (but it is constant over daytime and hence simpler) to get to the total noise of the system. The thermal noise of the system is not taken into account with this temperature and should be added by the generic noise adder. Note: the noise temperature plugged into the generic noise adder would tend to be lower, since it would then only simulate the noise of the system, but not the noise temperature seen by the antenna anymore.