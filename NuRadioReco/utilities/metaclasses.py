class Singleton(type):
    _instances = {}

    def __call__(cls, create_new=False, *args, **kwargs):
        if Singleton._instances.get(cls, None) is None or create_new:
            Singleton._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return Singleton._instances[cls]
