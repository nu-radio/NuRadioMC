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
        create_new: bool (default: None)
            * If ``False``, this will always attempt to return an existing instance of the class.
              If the existing instance was created with different initial arguments, this raises an error.
            * If ``True``, a new instance will always be created, even if one already exists.
            * If ``None`` (default), try to return an existing instance only if the provided arguments match,
              otherwise return a new instance (implies ``create_new=True``).

        Notes
        -----
        The `Singleton` metaclass is intended for classes that are expensive to initialize
        (e.g. because of a large amount of I/O, such as loading an antenna model from disk).
        """
        create_new = kwargs.pop('create_new', None)


        create_new = (
            create_new
            or cls not in Singleton._instances
            or create_new is None and (args, kwargs) != Singleton._args_kwargs[cls]
        )

        if create_new:
            logger.debug(f'Creating new instance of {cls}...')
            Singleton._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            Singleton._args_kwargs[cls] = (args, kwargs)
        elif (args, kwargs) != Singleton._args_kwargs[cls]:
            msg = (
                f'Singleton {cls} was already initialized with different arguments. '
                'To create a new instance with different initial values, include '
                '`create_new=True` in the class initialization call.'
                )
            raise ValueError(msg)

        return Singleton._instances[cls]
