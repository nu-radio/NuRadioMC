Modules
===========

In NuRadioReco, the process of reconstruction event properties from data is done
by modules. In principle, each step of an analysis should be done by a dedicated
module. For example, one module applies a filter to the channel spectrum and a
different module reconstructs the electric field from that channel afterwards.
In the case of simulation studies, modules can also be used to simulate aspects
of the detector response, e.g. the efieldToVoltageConverter, which calculates the
voltages in the antennas from the electric field at each antenna.

In principle, modules can be arranged in any order, including loops and if/or
branching, though some some modules may require others to be run beforehand. For
example, using a module to apply a filter to an electric field only makes sense
if the field has been reconstructed by another module beforehand.
It is even possible to combine several modules into a single module, for example
to make a standard reconstruction model, that performs a reconstruction with the
default settings deemed best for a given experiment.

Basic Module Structure
----------------------
Each module consists of four components:

* A constructor to create the module. In a reconstruction, this is usually
  called to create the modules before looping over all events in the event file.
* The ``begin`` method. This is used to specify settings that will not change on
  an event-by-event basis.
* The ``run`` method. This is executed for each event and executes the task that
  the module was built for.
* The ``end`` Method. This is run after the last event was processed for cleanup
  or to print information on what the module has done.

These methods should all be called in that order, though the *begin* and *end*
function can be skipped for some modules.

Logging
--------------
In version 2.3.0, the way in which we handle logging changed. We switched to a custom logging
class, which avoids any potential clashes with other packages. This custom class is set as the
logging class when using NuRadioMC/NuRadioReco. Furthermore, we now create
the NuRadioMC and NuRadioReco loggers automatically when importing the packages. This means
that no particular setup is required from the user, as long as the module loggers follow the
general naming scheme "NuRadioMC.MODULE" or "NuRadioReco.MODULE".

**Please note that here we only show examples from NuRadioReco.**
**For NuRadioMC applications one should simply replace NuRadioReco with NuRadioMC in the following.**

Logging in NuRadioReco is handled using a custom class called ``NuRadioLogger``,
which inherits from ``logging.Logger`` .
We use the fact that loggers can inherit handlers from logger created in parent
modules. To achieve this behaviour, every module file MODULE.py should initialise
a logger using

  .. code-block:: Python

    import logging

    logger = logging.getLogger('NuRadioReco.MODULE')
    logger.setLevel(logging.DEBUG)

The logger can then be used across the module, and will report using the name
given in the ``getLogger()`` function. Notice the logging level here. The convention
is to set the logging level to DEBUG, such that all messages are passed on to the parent
logger. Setting the logging level of a particular module to a higher value, will block
log messages lower than that level from that module. This can be useful when a particular
module produces a lot of logs that are not of interest when for example debugging a
different module.

The parent logger, with the name "NuRadioReco", is automatically initialised when importing NuRadioReco.
It is this parent logger which processes all the logging events from the NuRadioReco modules,
so it determines the overall logging level of the script that is inherited by all modules.
By default the level is set to STATUS (see below), but this can easily be changed by using
the ``set_general_log_level()`` function from the NuRadioReco logging module, located in
`NuRadioReco/utilities/logging.py` . This allows to turn DEBUG on for all, for example.
It is still possible to change the logging level for individual modules as an overwrite.
This can be used to block certain modules from passing messages on to the parent logger.
For example, the ``begin()`` function of the class MODULE could take parameter setting
the logging level:

  .. code-block:: Python

    class MODULE:
        def __init__(self):
            ...

        def begin(self, logging_level=None):
            if logging_level is not None:
                logger.setLevel(logging_level)

Next to the standard logging levels that Python provides, NuRadioReco implements
an additional level called STATUS. Its value is defined in `NuRadioReco/utilities/logging.py`
as LOGGING_STATUS. As of February 2024, this value is 25, which is between WARNING and
INFO (meaning that it will only be printed if the logging level is set to STATUS,
INFO or DEBUG). In order to log a message with the STATUS level, you can use the
``logger.status()`` function.

Another interesting feature of the Python logging module, is the option to add
multiple handlers which each output to a different location. Each of these can have
a different logging level. For example, if you wanted to only have the WARNING
output printed to the console, but at the same time save all the DEBUG statements to
a file for later reference, you can add a ``FileHandler`` to the general logger in your
script using

  .. code-block:: Python

    import logging
    # retrieve the parent logger
    logger = logging.getLogger("NuRadioReco")

    # create the handler, set the level and apply same format as parent logger
    f_handler = logging.FileHandler('debug.log')
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(logger.handlers[0].formatter)

    # add handler to the parent logger
    logger.addHandler(f_handler)

In the code block above we create a handler which will direct log statements to a file
called ``debug.log``, configure that handler to take all logging output with level DEBUG
or above and finally add that handler to the logger. Note here that the ``NuRadioLogger`
class is implemented such that adding a handler, automatically lowers the logging level
such that it is ensured the messages will indeed be passed on to the handler.
It also possible to set a custom formatter to this handler,
or add more handlers to the parent logger.
