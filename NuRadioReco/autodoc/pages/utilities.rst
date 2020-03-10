Utilities
============
*NuRadioReco* provides a number of utilities to make life easier, the most
important of which shall be introduced here.

Unit System
---------------------
To keep track of the units of the calculated event properties, *NuRadioReco*
offers its own units system. The usage is simple: For every input, the number
is multiplied by its units, for every output it is divided by it.

  .. code-block:: Python

    from NuRadioReco.utilities import units
    time = 132. * units.ms  # define 132 milli seconds
    d = 5. * units.mm   # define 5 mm
    v = d / time  # calculate speed
    print('The speed is {..2f} km/h'.format(v / (units.km / units.hour)))
    # the speed is 0.14 m / h

The advantage of this approach is, that the user does not need to worry about
(or even know) the units *NuRadioReco* uses internally. As long as all inputs
and outputs are multiplied and divided by the correct units, things will stay
consistent.

Fourier Transformation
--------------------------
Helper functions to perform fast Fourier Transform between time and frequency
domain. The functions are wrappers around the the numpy real Fourier transform
(since we know that signals are real in the time domain we can omit the negative
frequencies).
The spectrum is defined in volt/GHz instead of volt/bin, so that the values are
independent of the sampling rate.
  .. Important:: Always use these helper functions when doing Fourier transformations.
