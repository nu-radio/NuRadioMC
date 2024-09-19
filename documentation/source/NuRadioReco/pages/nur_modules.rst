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
**Please note that this section is specific to NuRadioReco.**
**For NuRadioMC applications one should replace NuRadioReco with NuRadioMC in the following.**
**Then, the** ``module.setup_logger()`` **function should be called with** ``name="NuRadioMC"``
**as a parameter, or** ``name=" "`` **when both NuRadioReco and NuRadioMC modules are used.**

Logging in NuRadioReco is handled using the standard Python logging interface.
It uses the fact that loggers can inherit handlers from logger created in parent
modules. To achieve this behaviour, every module file MODULE.py should initialise
a logger using

  .. code-block:: Python

    import logging

    logger = logging.getLogger('NuRadioReco.MODULE')

The logger can then be used across the module, and will report using the name
given in the ``getLogger()`` function.

Then any script using NuRadioReco modules should initialize a general logger with

  .. code-block:: Python

    import logging
    from NuRadioReco.utilities.logging import setup_logger
    logger = setup_logger()


This initializes a parent logger, which determines the overall logging level that is
inherited by all modules. By default the level is set to STATUS (see below), but this
can easily be changed by using the ``level`` keyword in the ``setup_logger()`` function.
This allows to turn DEBUG on for all, for example. It is still possible to change the
logging level for individual modules as an overwrite. For example, the ``begin()`` function
of the class MODULE could take parameter setting the logging level:

  .. code-block:: Python

    class MODULE:
        def __init__(self):
            ...

        def begin(self, logging_level=None):
            if logging_level is not None:
                logger.setLevel(logging_level)

Next to the standard logging levels that Python provides, NuRadioReco implements
an additional level called STATUS. Its value is defined in ``NuRadioReco/utilities/logging.py``
as LOGGING_STATUS. As of February 2024, this value is 25, which is between WARNING and
INFO (meaning that it will only be printed if the logging level is set to STATUS,
INFO or DEBUG). In order to log a message with the STATUS level, you can use the
``logger.status()`` function.

This behaviour is achieved using the ``addLoggingLevel()`` function implemented
in the ``NuRadioReco/utilities/logging.py`` module, which can also be used in
custom scripts to add additional logging levels together with corresponding logging
method to any logging class (this should also work with custom logging classes).

Another interesting feature of the Python logging module, is the option to add
multiple handlers which each output to a different location. Each of these can have
a different logging level. For example, if you wanted to only have the WARNING
output printed to the console, but at the same time save all the DEBUG statements to
a file for later reference, you can initialize the general logger in your script as

  .. code-block:: Python

    import logging
    from NuRadioReco.modules.base import module
    logger = module.setup_logger(level=logging.WARNING)

    f_handler = logging.FileHandler('debug.log')
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)

The last three lines in the code block above create a handler which will direct log
statements to a file called ``debug.log``, configure that handler to take all logging
output with level DEBUG or above and finally add that handler to the logger. It also
possible to set a custom formatter to this handler, or add more handler to the logger.
