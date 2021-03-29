from NuRadioReco.modules.base.module import register_run
import logging
logger = logging.getLogger('eventTypeIdentifier')


class eventTypeIdentifier:
    """
    Use this module to distinguish cosmic ray events from neutrino events
    """

    def begin(self):
        pass

    @register_run()
    def run(self, event, station, mode, forced_event_type='neutrino'):
        """
        Determines the type of event so that the correct modules can be
        executed accordingly. Currently neutrino and cosmic ray events are
        supported, but others can be added in the future.

        Parameters
        ---------------------
        event: event

        station: station

        mode: string
            specifies which criteria the decision to flag the event as cosmic ray
            or neutrino should be based on. Currently only supports 'forced',
            which automatically flags the event to a specified type. Add new
            modes to use different criteria
        forced_event_type: string
            if the mode is set to forced, this is the type the event will be set to.
            Currently supported options are neutrino and cosmic_ray
        """
        if mode == 'forced':
            if forced_event_type == 'neutrino':
                station.set_is_neutrino()
            elif forced_event_type == 'cosmic_ray':
                station.set_is_cosmic_ray()
            else:
                ValueError('Unsupported event type {}'.format(forced_event_type))
        else:
            raise ValueError('Unsupported mode {}'.format(mode))
