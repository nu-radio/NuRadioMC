"""
Module containing code to reconstruct the arrival directions of neutrinos.

The `voltageToEfieldAnalyticConverterForNeutrinos`
allows to reconstruct the arrival direction of a neutrino arriving
in an ARIANNA-style (shallow, LPDA-based) in-ice radio detector. The algorithm
is described in more detail in https://escholarship.org/uc/item/1bj9r6rb

The `neutrinoDirectionReconstructor`
module uses the deep (Vpol + Hpol) antennas to reconstruct the neutrino direction.
This algorithm is described in https://doi.org/10.1140/epjc/s10052-023-11604-w

"""