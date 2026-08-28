"""
Function to replace the njit decorator. If numba is not available we just use a dummy decorator, otherwise @njit
"""
try:
    from numba import njit
    print("Numba loaded")
except:
    print("Numba not available")
    def njit(*args, **kwargs):
        def dec(func):
            return func
        return dec