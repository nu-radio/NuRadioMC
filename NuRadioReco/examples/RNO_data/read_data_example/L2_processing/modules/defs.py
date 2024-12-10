import numpy as np

cvac = 0.3

def ior_exp1(z):
    
    # Note: z is given in natural feet, convert to meter
    def iorfunc(z):        
        A = 1.78
        B = 1.326
        C = 0.0202
        return A - (A - B) * np.exp(C * z * cvac)
    
    iorvals = iorfunc(z)
    iorvals[z > 0] = 1.0
    return iorvals

def ior_exp3(z):
    
    # Note: z is given in natural feet, convert to meter
    def iorfunc_snow(z):
        return 1.52737 - 0.298415 * np.exp(0.107158 * z * cvac)

    def iorfunc_firn(z):
        return 1.89275 - 0.521529 * np.exp(0.0136059 * z * cvac)

    def iorfunc_bubbly(z):
        return 1.77943 - 1.576 * np.exp(0.0403732 * z * cvac)

    z1 = -14.9 / cvac   # transition between snow and firn
    z2 = -80.5 / cvac   # transition between firn and bubbly ice
    
    snow_mask = np.argwhere(np.logical_and(z <= 0, z > z1))
    firn_mask = np.argwhere(np.logical_and(z <= z1, z > z2))
    bubbly_mask = np.argwhere(z <= z2)

    iorvals = np.zeros_like(z)    
    iorvals[snow_mask] = iorfunc_snow(z[snow_mask])
    iorvals[firn_mask] = iorfunc_firn(z[firn_mask])
    iorvals[bubbly_mask] = iorfunc_bubbly(z[bubbly_mask])
    iorvals[z > 0] = 1.0
    
    return iorvals
