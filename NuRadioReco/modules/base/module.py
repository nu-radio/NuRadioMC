from functools import wraps
from timeit import default_timer as timer
import NuRadioReco.framework.event
import NuRadioReco.framework.base_station
import NuRadioReco.detector.detector as detectors
import atexit
import inspect
import logging
import os
import pickle
from NuRadioReco.utilities.logging import LOGGING_STATUS

logger = logging.getLogger('NuRadioReco.module')

# Controls both recording and the printout at exit. Default from the environment variable NURADIO_TIMING (0 = off);
# can be changed at runtime via `NuRadioReco.modules.base.module.ENABLE_TIMING = False`.
ENABLE_TIMING = os.environ.get("NURADIO_TIMING", "1") not in ("", "0")
_timed_runs = []  # all decorated run methods, used for the timing summary


def setup_logger(name="NuRadioReco", level=None):
    """
    Set up the parent logger which all module loggers should pass their logs on to. Any handler which was
    previously added to the logger is cleared, and a single new `logging.StreamHandler()` with a custom
    formatter is added. Next to this, an extra logging level STATUS is added with level=`LOGGING_STATUS`,
    which is defined in `module.py` (as of February 2024, its value is 25). Then STATUS is also set as
    the default logging level.

    .. deprecated:: 2.3.0
                `module.setup_logger()` will be removed in v2.4.0 and replaced by
                `logging.setup_logger()` from the NuRadioReco utilities folder.

    Parameters
    ----------
    name : str, default="NuRadioReco"
        The name of the base logger
    level : int, default=25
        The logging level to use for the base logger
    """
    raise DeprecationWarning("Please update import, the logging module has moved to NuRadioReco.utilities.logging ."
                             "This function will be removed in v2.4.0 .")


_child_time = []  # stack: time spent in nested timed calls per active call, subtracted to get exclusive times
_order = {}  # module name -> index of first execution
_instances = {}  # id(module instance) -> instance, in order of first run() call


def _add_time(run_method, instance, duration, calls):
    # We use the module instance as key to time different module instances separately.
    if instance not in run_method.time:
        _order.setdefault(type(instance).__name__, len(_order))
    run_method.time[instance] = run_method.time.get(instance, 0) + duration
    run_method.calls[instance] = run_method.calls.get(instance, 0) + calls


def _timed_generator(gen, run_method, instance):
    """ Yields from `gen` and attributes the time spent producing each item (one call per item) to the module. """
    while True:
        _child_time.append(0.)
        start = timer()
        done = False
        try:
            item = next(gen)
        except StopIteration:
            done = True
        finally:
            duration = timer() - start
            nested = _child_time.pop()
            if _child_time:
                _child_time[-1] += duration
        _add_time(run_method, instance, duration - nested, 0 if done else 1)
        if done:
            return
        yield item


def register_run(level=None):
    """
    Decorator for run methods. This decorator registers the run methods. It allows to keep track of
    which module is executed in which order and with what parameters. Also the execution time of each
    module is tracked.
    """

    def run_decorator(run):

        # the signature is static, so inspect it only once (not per call)
        parameters = inspect.signature(run).parameters
        keys = [key for key in parameters.keys() if key != 'self']

        @wraps(run)
        def register_run_method(self, *args, **kwargs):

            # the following if/else part finds out if this module operates on full events or on a specific station
            # In principle, different modules can be executed on different stations, so we keep it general and save the
            # modules station specific.
            # The logic is: If the first two arguments are event and station -> station module
            # if the first argument is an event and the second not a station -> event module
            # if the first argument is not an event -> reader module that creates events. In this case, the module
            # returns an event and we use this event to store the module information (but the module actually returns a
            # generator, so not sure how to access the event.
            evt = None
            station = None

            _instances.setdefault(id(self), self)

            # convert args to kwargs to facilitate easier bookkeeping
            all_kwargs = {key: value for key, value in zip(keys, args)}
            # this silently overwrites positional args with kwargs, but this is probably okay as we still raise an error later
            all_kwargs.update(kwargs)

            # include parameters with default values
            for key, value in parameters.items():
                if key not in all_kwargs and value.default is not inspect.Parameter.empty:
                    all_kwargs[key] = value.default

            store_kwargs = {}
            for idx, (key, value) in enumerate(all_kwargs.items()):
                # event should be the first argument
                if isinstance(value, NuRadioReco.framework.event.Event) and idx == 0:
                    evt = value
                # station should be second argument
                elif isinstance(value, NuRadioReco.framework.base_station.BaseStation) and idx == 1:
                    station = value
                elif isinstance(value, (detectors.detector_base.DetectorBase, detectors.rnog_detector.Detector)):
                    pass  # we don't try to store detectors
                else:  # we try to store other arguments IF they are pickleable
                    try:
                        pickle.dumps(value, protocol=4)
                        store_kwargs[key] = value
                    except (TypeError, AttributeError):  # object couldn't be pickled - we store the error instead
                        store_kwargs[key] = TypeError(f"Argument of type {type(value)} could not be serialized")

            if station is not None:
                module_level = "station"
            elif evt is not None:
                module_level = "event"
            else:
                module_level = "reader"

            timing = ENABLE_TIMING
            if timing:
                _child_time.append(0.)
                start = timer()

            if module_level == "event":
                evt._register_module_event(self, self.__class__.__name__, store_kwargs)
            elif module_level == "station":
                evt._register_module_station(station.get_id(), self, self.__class__.__name__, store_kwargs)
            elif module_level == "reader":
                # not sure what to do... function returns generator, not sure how to access the event...
                pass

            try:
                res = run(self, *args, **kwargs)
            finally:
                if timing:
                    duration = timer() - start
                    nested = _child_time.pop()
                    if _child_time:
                        _child_time[-1] += duration

            if timing:
                is_generator = inspect.isgenerator(res)
                _add_time(register_run_method, self, duration - nested, 0 if is_generator else 1)
                if is_generator:
                    res = _timed_generator(res, register_run_method, self)

            return res

        register_run_method.time = {}
        register_run_method.calls = {}
        _timed_runs.append(register_run_method)

        return register_run_method

    return run_decorator


def end_all_modules():
    """
    Call `end()` of all module instances that have executed a decorated `run()`, in order of their first `run()` call.

    Each instance is ended only once. Exceptions in an `end()` are logged and do not stop the remaining ones.
    """
    instances = list(_instances.values())
    _instances.clear()
    for instance in instances:
        end = getattr(instance, "end", None)
        if callable(end):
            try:
                end()
            except Exception:
                logger.exception(f"Calling end() of {type(instance).__name__} failed")


def print_timing_summary():
    """
    Print the accumulated run time per module (class) in order of first execution.

    Times are exclusive: time spent in nested module calls (e.g. a generator wrapping another one) is
    attributed to the nested module only.

    For reader modules (generators), the time spent producing each item is counted, with one call per item.
    It is printed automatically at exit if ``ENABLE_TIMING`` is set (see top of file).
    """
    stats = {}
    for run in _timed_runs:
        for instance, total in run.time.items():
            entry = stats.setdefault(type(instance).__name__, [0., 0])
            entry[0] += total
            entry[1] += run.calls[instance]

    if not stats:
        return

    all_total = sum(t for t, _ in stats.values())
    rows = [f"{name:<40} {calls:>8} {total:>10.3f} {1e3 * total / calls:>14.3f} {100 * total / all_total:>6.1f}"
            for name, (total, calls) in sorted(stats.items(), key=lambda kv: _order[kv[0]])]
    header = f"{'module':<40} {'calls':>8} {'total [s]':>10} {'per call [ms]':>14} {'%':>6}"
    width = len(header)
    lines = ["┌" + "─" * (width + 2) + "┐", f"│ {header} │", "├" + "─" * (width + 2) + "┤"]
    lines += [f"│ {row} │" for row in rows]
    lines.append("└" + "─" * (width + 2) + "┘")
    logger.log(LOGGING_STATUS, "Module timing\n" + "\n".join("  " + line for line in lines))


def _print_timing_summary_at_exit():
    if ENABLE_TIMING:
        print_timing_summary()


atexit.register(_print_timing_summary_at_exit)
