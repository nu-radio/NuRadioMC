Reconstruction of the position of the position of the shower maximum Xmax
using the "LOFAR"-style analysis where the best CoREAS simulation is found
which matches the data best by comparing the energy fluence footprint.

The procedure is the following (following the description in S. Buitink Phys. Rev. D 90, 082003 https://doi.org/10.1103/PhysRevD.90.082003)
 * One star-shape CoREAS simulation is read in and interpolate to obtain the electric field
   at every antenna position. This is handled by the `readCoREASDetector` module
 * A full detector simulation is performed, i.e., simulation of antenna and signal chain
   response, adding of noise, electric field reconstruction (using standard unfolding), and
   calculation of the energy fluence at every antenna position
 * we loop over a set of CoREAS simulations (excluding the simulation we used to generate the data)
   of the same air-shower direction and energy and
   fit the fluence footprint to the simulated data to determine the core position and amplitude
   normalization factor. The chi^2 of the best fit is saved. 
 * the best fit chi^2 values are plotted against the true Xmax value from the simulations
 * a parabola is fitted to determine the most likely Xmax value


 A01createDetectorJson.py creastes the detector description
 A02CoREASFluenceXmaxReco.py downloads example CORSIKA7 simulations (verticla shower at the LOFAR site) and performs the fit