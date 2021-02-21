Use the Vertex Reconstruction Modules
======================================

NuRadioReco comes with 2 modules to reconstruct the position of the neutrino
interaction vertex. Both work in a very similar way: The expected difference
in signal arrival times between channels is determined for a possible vertex
position and the correlation between the channel waveforms is calculated for
that time shift. This is done for multiple channel pair and the correlations
for each pair are summed up. By scanning over all possible locations,the vertex
position can be determined by finding the point where the sum of correlations
is the largest.

Creating Lookup Tables
-------------------------

Redoing the raytracing for every point where the interaction vertex may be
will take too much time to be practical in the long run. Therefore, the propagation
times for a grid of positions is calculated once and stored to be used as a
lookup table later. To save computing time and storage space, cylindrical
symmetry of the signal propagation times is assumed, meaning that they do not
depend on the azimuth and can be stored as functions of the depth and the horizontal
distance from the antenna.

NuRadioReco provides a script to produce the lookup tables  at
`NuRadioReco/modules/neutrinoVertexReconstructor/create_lookup_tables.py`
