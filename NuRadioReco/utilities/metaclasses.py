class Singleton(type):
    """
    Can be assigned to classes as a metaclass.

    By default, only one instance of a Singleton can exist at a time,
    as the ``__call__`` method is overwritten to
    return the existing instance if one exists.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Overwrites the __call__ method

        Checks if an instance of the class already exists
        and returns that instance instead of creating a new one, unless
        ``create_new=True`` is specified.

        Parameters
        ----------
        create_new: bool (default:False)
            If set to true, a new instance will always be created, even if one already exists.
        """
        if 'create_new' in kwargs.keys():
            create_new = kwargs['create_new']
            kwargs.pop('create_new') # this kwarg should not be passed on to the class!
        else:
            create_new = False # the default
        if Singleton._instances.get(cls, None) is None or create_new:
            Singleton._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return Singleton._instances[cls]
