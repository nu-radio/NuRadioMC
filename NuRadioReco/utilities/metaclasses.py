class Singleton(type):
    """
    Can be assigned to classes as a metaclass.
    By default, only one instance of a Singleton can exist at a time, as the __call__ method is overwritten to
    return the existing instance if one exists.
    """
    _instances = {}

    def __call__(cls, create_new=False, *args, **kwargs):
        """
        Overwrites the __call__ method to check if an instance of the class already exists
        and returns that instance instead of creating a new one.

        Parameters
        ----------
        create_new: bool (default:False)
            If set to true, a new instance will always be created, event if one already exists.
        """
        if Singleton._instances.get(cls, None) is None or create_new:
            Singleton._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return Singleton._instances[cls]
