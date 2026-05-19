"""
IFT reconstruction script. should take event object and do pulse-finding, outlier cleaning,
noise modelig and reconstruct. Post-reconstruction should update event parameters
and optionally export full posterior samples and plot.

needed new modules:
    - LOFAR.utilities.iftPulseFinder (pulse finding and outlier cleaning)
    - LOFAR.utilities.iftModel (forward model for LOFAR)
"""