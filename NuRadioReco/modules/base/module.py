from functools import wraps
from timeit import default_timer as timer
import NuRadioReco.framework.event
import NuRadioReco.framework.base_station


def register_run(level=None):
    if level not in ["station", "event"]:
        raise NotImplementedError("The level needs to be either 'station' or 'event'")

    def run_decorator(run):

        @wraps(run)
        def register_run_method(self, *args, **kwargs):

            evt = None
            station = None
            level = None
            # find out type of module automatically
            if(len(args) == 1):
                if(isinstance(args[0], NuRadioReco.framework.event.Event)):
                    level = "event"
                    evt = args[0]
                else:
                    raise AttributeError("first argument of run method is not of type NuRadioReco.framework.event.Event")
            elif(len(args) >= 2):
                if(isinstance(args[0], NuRadioReco.framework.event.Event) and isinstance(args[1], NuRadioReco.framework.base_station.BaseStation)):
                    level = "station"
                    evt = args[0]
                    station = args[1]
                elif(isinstance(args[0], NuRadioReco.framework.event.Event)):
                    level = "station"
                    evt = args[0]
                else:
                    raise AttributeError("first argument of run method is not of type NuRadioReco.framework.event.Event")
            else:
                # this is a module that creats events, not sure how to register such modules because an event is not yet available when the module is called
                pass
#                 raise AttributeError("run method has no argument")

            if(level == "event"):
                evt.register_module_event(self, self.__class__.__name__, kwargs)
            elif(level == "station"):
                evt.register_module_station(station.get_id(), self, self.__class__.__name__, kwargs)
            start = timer()
            res = run(self, *args, **kwargs)
            end = timer()
            if not self in register_run_method.time:
                register_run_method.time[self] = 0
            register_run_method.time[self] += (end - start)
            return res

        register_run_method.time = {}

        return register_run_method

    return run_decorator
