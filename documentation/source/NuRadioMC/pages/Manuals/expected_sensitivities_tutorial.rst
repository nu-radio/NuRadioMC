Example: Calculating expected sensitivities
===========================================

This page describes how to calculate the expected sensitivity from a MC simulation. 


The equations to calculate the effective volume from the simulation output and how to convert this into
expected limits are summarized here: https://github.com/nu-radio/NuRadioMC/raw/master/NuRadioMC/doc/effective_volume_calculation.pdf


NuRadioMC already implements all required functions:

The :mod:`NuRadioReco.utilities.Veff <NuRadioMC.utilities.Veff>` utility script 
allows to calculate the effective volume from the simulation output. 
The :func:`get_Veff <NuRadioMC.utilities.Veff.get_Veff_Aeff>` function will be needed by most people. 

The effective volume can be converted into an expected sensitivity with
:func:`get_limit_e2_flux <NuRadioMC.utilities.fluxes.get_limit_e2_flux>`. The example module
`NuRadioMC/examples/Sensitivities/limit_plot.py <https://github.com/nu-radio/NuRadioMC/blob/master/NuRadioMC/examples/Sensitivities/limit_plot.py>`_
draws the 'standard' sensitivity plot with published limits and flux models and adds your sensitivity to it:

.. code-block:: Python

    from NuRadioMC.examples.Sensitivities import limit_plot
    from NuRadioReco.utilities import units
    from matplotlib import pyplot as plt
    fig, ax = limit_plot.get_limit_figure(diffuse=True, show_grand_10k=False, show_grand_200k=False, show_Heinze=True, show_TA=True, show_ara=False, show_arianna=False)
    labels = limit_plot.add_limit(ax, [], energies, Veff * 4 * np.pi, n_stations=200, label='my limit', livetime=5 * units.year, linestyle='-', color='blue', linewidth=3)
    plt.legend(handles=labels, loc=2)

Run ``python limit_plot.py --help`` to make the same plot from an effective volume file on the command line.
The flux models (cosmogenic, astrophysical and source populations) are available as all-flavour fluxes in
:mod:`NuRadioMC.utilities.neutrino_fluxes`.
