from functools import wraps
from timeit import default_timer as timer
import NuRadioReco.framework.event
import NuRadioReco.framework.base_station
import logging


def setup_logger(name="NuRadioReco", level=logging.WARNING):

    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('\033[93m%(levelname)s - \033[0m%(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger



def register_run(level=None):
    """
    Decorator for run methods. This decorator registers the run methods. It allows to keep track of
    which module is executed in which order and with what parameters. Also the execution time of each
    module is tracked.
    """

    def run_decorator(run):

        @wraps(run)
        def register_run_method(self, *args, **kwargs):

            # the following if/else part finds out if this is a module that operates on full events or a specific station
            # In principle, different modules can be executed on different stations, so we keep it general and save the
            # modules station specific.
            # The logic is: If the first two arguments are event and station -> station module
            # if the first argument is an event and the second not a station -> event module
            # if the first argument is not an event -> reader module that creates events. In this case, the module
            # returns an event and we use this event to store the module information (but the module actually returns a
            # generator, so not sure how to access the event.
            evt = None
            station = None
            level = None
            # find out type of module automatically
            if(len(args) == 1):
                if(isinstance(args[0], NuRadioReco.framework.event.Event)):
                    level = "event"
                    evt = args[0]
                else:
                    # this is a module that creats events
                    level = "reader"
            elif(len(args) >= 2):
                if(isinstance(args[0], NuRadioReco.framework.event.Event) and isinstance(args[1], NuRadioReco.framework.base_station.BaseStation)):
                    level = "station"
                    evt = args[0]
                    station = args[1]
                elif(isinstance(args[0], NuRadioReco.framework.event.Event)):
                    level = "event"
                    evt = args[0]
                else:
                    # this is a module that creats events
                    level = "reader"
                    raise AttributeError("first argument of run method is not of type NuRadioReco.framework.event.Event")
            else:
                # this is a module that creats events
                level = "reader"

            start = timer()
            res = run(self, *args, **kwargs)
            if(level == "event"):
                evt.register_module_event(self, self.__class__.__name__, kwargs)
            elif(level == "station"):
                evt.register_module_station(station.get_id(), self, self.__class__.__name__, kwargs)
            elif(level == "reader"):
                # not sure what to do... function returns generator, not sure how to access the event...
                pass
            end = timer()
            if not self in register_run_method.time:  # keep track of timing of modules. We use the module instance as key to time different module instances separately.
                register_run_method.time[self] = 0
            register_run_method.time[self] += (end - start)
            return res

        register_run_method.time = {}

        return register_run_method

    return run_decorator
