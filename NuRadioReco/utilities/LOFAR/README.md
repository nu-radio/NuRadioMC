## LOFAR utilities

Here, all utilites used for the LOFAR reconstruction pipeline is included.

NOTE: all JAX- and NIFTY-related imports are in a try-except clause to avoid any import errors for NuRadio users without jax / nifty. In the future, or someplace else, there should be warning / exception to install nifty & JAX if not installed.

All macros (i.e. fixed values, paths to directories) that are set for the pipeline are stored here in macros.py. In principle we can move this elsewhere (e.g. in the pipeline/ directory) as well. 