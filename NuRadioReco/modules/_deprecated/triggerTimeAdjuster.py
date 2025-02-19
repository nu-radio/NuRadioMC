import warnings

class triggerTimeAdjuster:

    def __init__(self, *args, **kwargs):
        warnings.error("The module `triggerTimeAdjuster `is deprecated. In most cased you can safely delete the application "
                       "of this module as it is automatically applied in NuRadioMC simulations. If you really need to use this module, "
                       "please use the channelReadoutWindowCutter module instead.", DeprecationWarning)
        raise NotImplementedError
