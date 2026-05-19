"""
Main wrapper for the LOFAR pipeline. Calls several modules in sequence to process an event:
    -modules.io.LOFAR.readLOFARData
    -modules.LOFAR.eventTypeIdentified
    -modules.LOFAR.rfi_filter
    -modules.LOFAR.calibrator
    -modules.LOFAR.bandpassFilter
    -modules.LOFAR.pulseFinder
    -modules.LOFAR.planWaveFitter
    -modules.LOFAR.voltageToEfield
    -modules.LOFAR.iftReconstructor
    -modules.LOFAR.visualizer
    -some sort of event writing (.nur File without traces)
"""