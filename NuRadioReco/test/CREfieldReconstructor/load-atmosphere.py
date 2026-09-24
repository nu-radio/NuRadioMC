"""Script that triggers loading of radiotools atmosphere

Loading an atmosphere that is not downloaded previously
will call sys.exit on older radiotools versions without
raising an error.
"""
import radiotools.atmosphere.models

if __name__ == "__main__":
    atm = radiotools.atmosphere.models.Atmosphere()