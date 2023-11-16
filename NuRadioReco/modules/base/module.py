from functools import wraps
from timeit import default_timer as timer
import NuRadioReco.framework.event
import NuRadioReco.framework.base_station
import NuRadioReco.detector.detector_base
import logging
import inspect
import pickle

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

            signature = inspect.signature(run)
            parameters = signature.parameters
            # convert args to kwargs to facilitate easier bookkeeping
            keys = [key for key in parameters.keys() if key != 'self']
            all_kwargs = {key:value for key,value in zip(keys, args)}
            all_kwargs.update(kwargs) # this silently overwrites positional args with kwargs, but this is probably okay as we still raise an error later

            # include parameters with default values
            for key,value in parameters.items():
                if key not in all_kwargs.keys():
                    if value.default is not inspect.Parameter.empty:
                        all_kwargs[key] = value.default

            store_kwargs = {}
            for idx, (key,value) in enumerate(all_kwargs.items()):
                if isinstance(value, NuRadioReco.framework.event.Event) and idx == 0: # event should be the first argument
                    evt = value
                elif isinstance(value, NuRadioReco.framework.base_station.BaseStation) and idx == 1: # station should be second argument
                    station = value
                elif isinstance(value, NuRadioReco.detector.detector_base.DetectorBase):
                    pass # we don't try to store detectors
                else: # we try to store other arguments IF they are pickleable
                    try:
                        pickle.dumps(value, protocol=4)
                        store_kwargs[key] = value
                    except (TypeError, AttributeError): # object couldn't be pickled - we store the error instead
                        store_kwargs[key] = TypeError(f"Argument of type {type(value)} could not be serialized")
            if station is not None:
                module_level = "station"
            elif evt is not None:
                module_level = "event"
            else:
                module_level = "reader"

            start = timer()

            if module_level == "event":
                evt.register_module_event(self, self.__class__.__name__, store_kwargs)
            elif module_level == "station":
                evt.register_module_station(station.get_id(), self, self.__class__.__name__, store_kwargs)
            elif module_level == "reader":
                # not sure what to do... function returns generator, not sure how to access the event...
                pass

            res = run(self, *args, **kwargs)

            end = timer()

            if self not in register_run_method.time:  # keep track of timing of modules. We use the module instance as key to time different module instances separately.
                register_run_method.time[self] = 0
            register_run_method.time[self] += (end - start)

            return res

        register_run_method.time = {}

        return register_run_method

    return run_decorator
