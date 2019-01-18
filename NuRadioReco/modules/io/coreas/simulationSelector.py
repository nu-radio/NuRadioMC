import numpy as np
import logging

logger = logging.getLogger('simulationSelector')

class simulationSelector:
    '''
    Module that let's you select CoREAS simulations
    based on certain criteria, e.g. signal in a relevant band
    certain arrival directions, energies, etc.
    '''

    def __init__(self):
        self.__t = 0
        self.begin()

    def begin(self, debug=False):
        pass

    def run(self, evt):
        t = time.time()


        self.__t += time.time() - t

    def end(self):
        from datetime import timedelta
        logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        logger.info("total time used by this module is {}".format(dt))
        return dt