from NuRadioReco.modules.base.module import register_run
import time
import logging
logger = logging.getLogger('efieldAirToIcePropagator')


class efieldAirToIcePropagator:
    """
    Module that propagates the efield (usually from air showers) into the ice.
    """

    def __init__(self):
        self.__t = 0
        self.__debug = None
        self.begin()

    def begin(self, debug=False):
        self.__debug = debug

    @register_run()
    def run(self, evt, station, det):
        t = time.time()

        logger.warning("Nothing implemented yet")

        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt
