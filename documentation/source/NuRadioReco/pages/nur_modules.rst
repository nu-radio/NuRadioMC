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

.. tip::
    If you are writing a module for a specific experiment, it is recommended to follow the naming
    scheme "NuRadioReco.EXPERIMENT.MODULE" to avoid any potential clashes with other experiments.
    It also allows to adjust the logging level for all modules of a specific experiment at once,
    by manually the settings of the logger "NuRadioReco.EXPERIMENT".

.. note::
    Please note that here we only show examples from NuRadioReco.
    For NuRadioMC applications one should simply replace NuRadioReco with NuRadioMC in the following.

Logging in NuRadioReco is handled using a custom class called ``NuRadioLogger``,
which inherits from ``logging.Logger`` .
We use the fact that loggers can inherit handlers from logger created in parent
modules. To achieve this behaviour, every module file MODULE.py should initialise
a logger using

  .. code-block:: Python

    import logging

    logger = logging.getLogger('NuRadioReco.MODULE')

The logger can then be used across the module and will report using the name
given in the ``getLogger()`` function. Notice that we **do not set the logging level**
**explicitly in the module**. This ensures that the logging level is inherited from the
parent logger, which allows users to easily control the logging level of all modules at once.
If you want to provide the functionality to change the logging level of the module,
for example in the ``begin()`` function, please make sure the default is `logging.NOTSET`.

  .. code-block:: Python

    logger = logging.getLogger('NuRadioReco.MODULE')

    class MODULE:
        def __init__(self):
            self.logger = logger

        def begin(self, logging_level=logging.NOTSET):
            self.logger.setLevel(logging_level)

The parent logger, with the name "NuRadioReco", is automatically initialised when importing NuRadioReco.
It is this parent logger which processes all the logging events from the NuRadioReco modules,
and it determines the overall logging level of the script that is inherited by all modules.
By default the level is set to STATUS (see below), but this can easily be changed by using
the ``set_general_log_level()`` function from the NuRadioReco logging module, located in
`NuRadioReco/utilities/logging.py` . This allows to turn DEBUG on for all, for example.
It is still possible to change the logging level for individual modules as an overwrite.
This can be useful when debugging a particular module, as in the following example.
Here we first set the general logging level to ERROR and then use the ``begin()`` function
to set the logging level of the module to DEBUG. This will show DEBUG messages for this module,
while only showing ERROR messages for all other modules.

  .. code-block:: Python

    from NuRadioReco.utilities.logging import set_general_log_level
    import logging

    set_general_log_level(logging.ERROR)  # set all modules logging level to ERROR

    module = MODULE()
    module.begin(logging_level=logging.DEBUG)  # show DEBUG messages for this module

Next to the standard logging levels that Python provides, NuRadioReco implements
an additional level called STATUS. Its value is defined in `NuRadioReco/utilities/logging.py`
as LOGGING_STATUS. As of February 2024, this value is 25, which is between WARNING and
INFO (meaning that it will only be printed if the logging level is set to STATUS,
INFO or DEBUG). In order to log a message with the STATUS level, you can use the
``my_logger.status()`` method.

Another interesting feature of the Python logging module, is the option to add
multiple handlers which each output to a different location. Each of these can have
a different logging level and can be formatted differently. To apply the same formatting
as the NuRadioReco logger, you can use the ``get_fancy_formatter()`` function from the
`NuRadioReco/utilities/logging.py` module. Regarding the logging level, please note that
the logger's level will take priority (this `diagram <https://docs.python.org/3/howto/logging.html#logging-flow>`_
shows the control flow).

For example, if you wanted to save all the logging statements to a file for later reference,
you can add a ``FileHandler`` to the general logger in your script using

  .. code-block:: Python

    from NuRadioReco.utilities.logging import get_fancy_formatter, set_general_log_level
    import logging

    set_general_log_level(logging.DEBUG)  # set all modules logging level to DEBUG

    # create the handler, set the level and apply same formatting as NuRadioReco logger
    f_handler = logging.FileHandler('debug.log')
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(get_fancy_formatter())

    # retrieve parent logger and add file handler to it
    logger = logging.getLogger("NuRadioReco")
    logger.addHandler(f_handler)

In the code block above we create a handler which will direct log statements to a file
called ``debug.log``, configure that handler to take all logging output with level DEBUG
or above and format them using the same formatting as the NuRadioReco logger. We then
add that handler to the NuRadioReco logger. Notice that we also had to set the general
logging level to DEBUG, to ensure that DEBUG messages will actually be logged. Another way
could be to set the logging level of some specific modules to DEBUG. It also possible to
set a different formatter to this handler, or add more handlers to the parent logger.

.. hint::
    If you wish to log records with a level lower than the general logging level to a file,
    you can edit the ``StreamHandler`` of the NuRadioReco logger which is created by default
    (you should be able to retrieve this one via ``parent_logger.handlers[0]``).
    The idea would be to set the logger's level as low as possible, but up the level of the
    ``StreamHandler`` to only output messages of a certain level or higher. The level of the
    ``FileHandler`` would then be set to the lowest level you want to log to the file. Note
    that this might mess up the standard logging behaviour, so only do this when you know what
    you are doing.
