from __future__ import absolute_import, division, print_function
import numpy as np
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
import radiopropa

def get_ice_model(name):
    ice = medium.get_ice_model(name)
    air_boundary = radiopropa.Discontinuity(radiopropa.Plane(radiopropa.Vector3d(0,0,0), radiopropa.Vector3d(0,0,1)), ice.get_index_of_refraction(np.array([0,0,0])), 1)
    if(name == "ARAsim_southpole"):
        ice_model = radiopropa.IceModel_Exponential(z_surface=0, n_ice=ice.n_ice, delta_n=ice.delta_n, z_0=ice.z_0*radiopropa.meter/units.meter)
        discontinuities = {"air_boundary":air_boundary}
        observers = {}
    elif(name == "southpole_simple"):
        ice_model = radiopropa.IceModel_Exponential(z_surface=0, n_ice=ice.n_ice, delta_n=ice.delta_n, z_0=ice.z_0*radiopropa.meter/units.meter)
        discontinuities = {"air_boundary":air_boundary}
        observers = {}
    elif(name == "southpole_2015"):
        ice_model = radiopropa.IceModel_Exponential(z_surface=0, n_ice=ice.n_ice, delta_n=ice.delta_n, z_0=ice.z_0*radiopropa.meter/units.meter)
        discontinuities = {"air_boundary":air_boundary}
        observers = {}
    elif(name == "mooresbay_simple"):
        ice_model = radiopropa.IceModel_Exponential(z_surface=0, n_ice=ice.n_ice, delta_n=ice.delta_n, z_0=ice.z_0*radiopropa.meter/units.meter)
        reflection_bottom = radiopropa.ReflectiveLayer(radiopropa.Plane(radiopropa.Vector3d(0,0,ice.reflection*radiopropa.meter/units.meter),
                                                        radiopropa.Vector3d(0,0,1)),ice.reflection_coefficient)
        discontinuities = {"air_boundary":air_boundary,"bottom_reflection":reflection_bottom}
        observers = {}
    elif(name == "greenland_simple"):
        ice_model = radiopropa.IceModel_Exponential(z_surface=0, n_ice=ice.n_ice, delta_n=ice.delta_n, z_0=ice.z_0*radiopropa.meter/units.meter)
        discontinuities = {"air_boundary":air_boundary}
        observers = {}
    else:
        raise NotImplementedError('The requested ice model has no numerical implementation yet')
    	
    return {"ice_model": ice_model, "discontinuities": discontinuities, "observers": observers}