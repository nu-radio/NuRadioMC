class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if Singleton._instances.get(cls, None) is None:
            Singleton._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return Singleton._instances[cls]
