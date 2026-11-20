def __getattr__(name):
    # E2_fluxes3 was split up; give users of the old module a pointer to its replacements
    if name == "E2_fluxes3":
        raise ImportError(
            "NuRadioMC.examples.Sensitivities.E2_fluxes3 was removed. Use "
            "NuRadioMC.examples.Sensitivities.limit_plot (get_limit_figure, add_limit) for limit plots and "
            "NuRadioMC.utilities.neutrino_fluxes for the neutrino flux models.")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
