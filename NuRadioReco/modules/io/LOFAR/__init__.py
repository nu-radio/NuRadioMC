"""
Package with modules related to the read-in of LOFAR (.root) data files

Most users will only be interested in the `readLOFARData.readLOFARData` class,
which directly converts the raw LOFAR data to the NuRadioReco `Event <NuRadioReco.framework.event.Event>`
structure. The modules `rawTBBio`, `rawTBBio_metadata` and `rawTBBio_utilities` contain several
helper functions and classes that enable this, but it shouldn't generally be necessary to interact
with these if you're just trying to read in LOFAR data.

"""
