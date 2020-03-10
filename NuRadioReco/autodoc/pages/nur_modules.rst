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
It is event possible, to combine several modules into a single module, for example
to make a standard reconstruction model, that performs a reconstruction with the
default settings deemed best for a given experiment.

Basic Module Structure
-----------
Each module consists of four components:
  * A constructor to create the module. In a reconstruction, this is usually
  called to create the modules before looping over all events in the event file.

  * The *begin* method. This is used to specify settings that will not change on
  an event-by-event basis.

  * The *run* method. This is executed for each event and executes the task that
  the module was built for.

  * The *end* Method. This is run after the last event was processed for cleanup
  or to print information on what the module has done.

These methods should all be called in that order, though the *begin* and *end*
function can be skipped for some modules.

Logging
--------------
  Every module MODULE should have in the init function the initalization of a
  logger with

  .. code-block:: Python

    self.logger = logging.getLogger('NuRadioReco.MODULE')


  This expects that the script containing the module sequence initalized a
  general logger with

  .. code-block:: Python

    import logging
    from NuRadioReco.modules.base import module
    logger = module.setup_logger(level=logging.WARNING)


  This initalizes a parent logger, which determines the overall logging level that is inherited by all modules. This allows to turn DEBUG on for all, for example. It is still possible to change the logging level for individual modules as an overwrite.
