"""
Package with modules related to the read-in of LOFAR (.h5) data files

Most users will only be interested in the `readLOFARData.readLOFARData` class,
which directly converts the raw LOFAR data to the NuRadioReco
`Event <NuRadioReco.framework.event.Event>` structure.

The ``_rawTBBio`` modules contain several helper functions and classes that enable this,
which are adapted from https://github.com/Bhare8972/LOFAR-LIM/tree/master/LoLIM/IO.
It shouldn't generally be necessary to interact with these directly
if you're just trying to read in LOFAR data.

"""
