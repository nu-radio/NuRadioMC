from NuRadioMC.utilities.medium_base import*
       

class southpole_simple(IceModel_Exponential):
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 RICE2014/SP model
        # define model parameters (RICE 2014/southpole)
        IceModel_Exponential.__init__(self, z_bottom = -2500*units.meter, 
                                        n_ice = 1.78, z_0 = 71.*units.meter, 
                                        delta_n = 0.426)


class southpole_2015(IceModel_Exponential):
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 SPICE2015/SP model
        IceModel_Exponential.__init__(self, z_bottom = -2500*units.meter, 
                                        n_ice = 1.78, z_0 = 77.*units.meter, 
                                        delta_n = 0.423)


class ARAsim_southpole(IceModel_Exponential):
    def __init__(self):
        # define model parameters (SPICE 2015/southpole)
        IceModel_Exponential.__init__(self, z_bottom = -2500*units.meter, 
                                        n_ice = 1.78, z_0 = 75.75757575757576*units.meter, 
                                        delta_n = 0.43)


class mooresbay_simple(IceModel_Exponential,IceModel_ReflectiveBottom):
    def __init__(self):
        # from https://doi.org/10.3189/2015JoG14J214
        IceModel_ReflectiveBottom.__init__(self, z_refl = -576*units.m, 
                                            refl_coef = 0.82, refl_phase_shift = 180*units.deg)

        # from https://doi.org/10.1088/1475-7516/2018/07/055 MB1 model
        IceModel_Exponential.__init__(self, z_bottom = self.reflection-1*units.m, 
                                        n_ice = 1.78, z_0 = 34.5*units.meter, 
                                        delta_n = 0.46)


class mooresbay_simple_2(IceModel_Exponential):
    def __init__(self):\
        # from https://doi.org/10.3189/2015JoG14J214
        IceModel_ReflectiveBottom.__init__(self,z_refl = -576*units.m,
                                            refl_coef = 0.82,refl_phase_shift = 180*units.deg)

        # from https://doi.org/10.1088/1475-7516/2018/07/055 MB2 model
        IceModel_Exponential.__init__(self,z_bottom = self.reflection-1*units.m, 
                                        n_ice = 1.78, z_0 = 37*units.meter, 
                                        delta_n = 0.481)


class greenland_simple(IceModel_Exponential):
    def __init__(self):
        # from C. Deaconu, fit to data from Hawley '08, Alley '88
        # rho(z) = 917 - 602 * exp (-z/37.25), using n = 1 + 0.78 rho(z)/rho_0
        IceModel_Exponential.__init__(self,z_bottom = -3000*units.meter, 
                                        n_ice = 1.78, z_0 = 37.25*units.meter, 
                                        delta_n = 0.51)





def get_ice_model(name):
    if globals()[name]() == None:
        logger.error('The ice model you are trying to use is not implemented. Please choose another ice model or implement a new one.')
        raise NotImplementedError('The ice model you are trying to use is not implemented. Please choose another ice model or implement a new one.')
    else:
        return globals()[name]()