import logging
logger = logging.getLogger('NuRadioReco.metaclasses')

class Singleton(type):
    """
    Can be assigned to classes as a metaclass.

    By default, only one instance of a Singleton can exist at a time,
    as the ``__call__`` method is overwritten to
    return the existing instance if one exists.
    """
    _instances = {}
    _args_kwargs = {}

    def __call__(cls, *args, **kwargs):
        """
        Overwrites the __call__ method

        Checks if an instance of the class already exists
        and returns that instance instead of creating a new one, unless
        ``create_new=True`` is specified.

        Parameters
        ----------
        create_new: bool (default: False)
            If set to true, a new instance will always be created, even if one already exists.
        """
        create_new = kwargs.pop('create_new', False)

        if Singleton._instances.get(cls, None) is None or create_new:
            Singleton._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            Singleton._args_kwargs[cls] = (args, kwargs)
        elif (args, kwargs) != Singleton._args_kwargs[cls]:
            logger.warning(
                f'{cls} is a Singleton, returning existing instance and ignoring '
                'arguments passed to __init__. To create a new instance instead, include '
                '`create_new=True` in the class initialization call.'
                )

        return Singleton._instances[cls]
