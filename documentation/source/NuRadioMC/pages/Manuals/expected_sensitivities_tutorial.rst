Example: Calculating expected sensitivities
===========================================

This page describes how to calculate the expected sensitivity from a MC simulation. 


The equations to calculate the effective volume from the simulation output and how to convert this into
expected limits are summarized here: https://github.com/nu-radio/NuRadioMC/raw/master/NuRadioMC/doc/effective_volume_calculation.pdf


NuRadioMC already implements all required functions:

The :mod:`NuRadioReco.utilities.Veff <NuRadioMC.utilities.Veff>` utility script 
allows to calculate the effective volume from the simulation output. 
The :func:`get_Veff <NuRadioMC.utilities.Veff.get_Veff_Aeff>` function will be needed by most people. 

The effective volume can be converted into an expected sensitivity using 
the example script `NuRadioMC/examples/Sensitivities/E2_fluxes3.py <https://github.com/nu-radio/NuRadioMC/blob/master/NuRadioMC/examples/Sensitivities/E2_fluxes3.py>`_.
An example how is can be used to create the 'standard' sensitivity plot is given below:

.. code-block:: Python

    from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limits
    from NuRadioMC.utilities import units
    from matplotlib import pyplot as plt
    fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=False, show_grand_200k=False, show_Heinze=True, show_TA=True, show_ara=False, show_arianna=False)
    limits.add_limit(ax, labels, energies, Veff * 4 * np.pi, n_stations=200, label='my limit', livetime=5 * units.year, linestyle='-', color='blue', linewidth=3) 

