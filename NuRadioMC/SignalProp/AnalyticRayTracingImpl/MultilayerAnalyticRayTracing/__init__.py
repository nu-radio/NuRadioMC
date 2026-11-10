"""
Implementation of the multi-layer analytic raytracer

See https://arxiv.org/abs/2609.05240 for more details

Submodules:
-----------
:mod:`.corefunctions`
    internal functions
:mod:`.getrayparameters`
    functions to get ray parameters (angles, path lengths, launch vectors...)
:mod:`.multilayeranalyticraytracing`
    Implementation of the 2D ray tracer class
:mod:`.planewave`
    Function to compute timing differences for a plane-wave source
:mod:`.solver`
    Functions to run the solver
:mod:`.testutils`
    Tools to help test / compare different ray tracers

"""