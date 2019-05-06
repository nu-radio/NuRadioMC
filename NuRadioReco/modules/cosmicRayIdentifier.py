import logging
logger = logging.getLogger('cosmicRayIdentifier')


class cosmicRayIdentifier:
    """
    Use this module to distinguish cosmic ray events from neutrino events
    """

    def begin(self):
        pass

    def run(self, event, station, mode):
        """
        Decides if event recorded by station should be treated as a cosmic ray
        event and sets the is_neutrino flag accordingly

        Parameters
        ---------------------
        event: event

        station: station

        mode: string
            specifies which criteria the decision to flag the event as cosmic ray
            should be based on. Currently only supports 'forced', which automatically
            flags the event as cosmic ray. Add new modes to use different criteria
        """

        if mode == 'forced':
            if station.is_cosmic_ray():
                logger.warning('Event is already flagged as cosmic ray.')
            station.set_is_cosmic_ray()
        else:
            raise ValueError('Unsupported mode {}'.format(mode))


