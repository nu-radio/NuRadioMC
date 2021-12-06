Antenna Models
=========================================

Overview of available Antenna Models
--------------------------------------

The following antenna models are available in NuRadioReco.
The headings are the unique identifiers of the antenna model that need to specified as  ``antenna_type`` in the detector description.

All antenna models are stored on a central data server and are downloaded automatically on-demand
whenever the user requests the antenna model for the first time.

For developers:
If you add new an antenna model please add the sha1sum = hashlib.sha1() to this list and send Christian
the antenna model so that he can put it on our central server.

Inf means usually an infinite medium, firn has a refractive index n = 1.3-1.4. Since this is a bit imprecise
we changed the naming to the actual refractive index.

bicone_v8_infAir
-----------------
WIPL-D simulation of ARA Bicone antenna.
This antenna has been used by ARIANNA at the South Pole.
The antenna is embedded in an infinite medium with an index of refraction of n = 1.0.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

bicone_v8_inf_n1.32
--------------------
WIPL-D simulation of ARA Bicone antenna.
This antenna has been used by ARIANNA at the South Pole.
The antenna is embedded in an infinite medium with an index of refraction of n = 1.32.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

bicone_v8_inf_n1.4
-------------------
WIPL-D simulation of ARA Bicone antenna.
This antenna has been used by ARIANNA at the South Pole.
The antenna is embedded in an infinite medium with an index of refraction of n = 1.4.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

bicone_v8_inf_n1.78
--------------------
WIPL-D simulation of ARA Bicone antenna.
This antenna has been used by ARIANNA at the South Pole.
The antenna is embedded in an infinite medium with an index of refraction of n = 1.78.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_InfFirn
--------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
The antenna is embedded in an infinite medium with an index of refraction of n = 1.3.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz assumed

Last updated: 2018

createLPDA_InfFirn_n1.4
--------------------------
Same as createLPDA_100MHz_InfFirn but antenna embedded in infinite firn with index of n = 1.4.
Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z1cm_InFirn_RG
----------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
lowest/largest tine 1cm above air (this is because in the simulation the geometry is inverted, the ground is air and the medium of the antenna is firn).
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_z1cm_InFirn_BoresightToBoundary
-------------------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
smallest/highest tine 1cm below air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

createLPDA_z10cm_InFirn_RG
--------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
lowest/largest tine 10cm above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_z1m_InFirn_RG
------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
lowest/largest tine 1m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_z2m_InFirn_RG
-------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. Largest tine 2m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_z2m_InFirn_Backlobe_NoRG
------------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. Largest tine 2m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [200,1000]MHz

Last updated: 2018

createLPDA_z3m_InAir_RG
------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. lowest/largest tine 3m above firn.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_z3m_InFirn_BoresightToBoundary
-------------------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
Largest tine 3m below air; nose 1.58m below air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

createLPDA_z3mAndLPDALen_InFirn_BoresightToBoundary
---------------------------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. Nose 3.2m below air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_z5m_InFirn_RG
-------------------------
WIPL-D simulation of 100 MHz LPDA from create. This antenna is used by ARIANNA.
Largest tine 5m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_z10m_InFirn_RG
--------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
Largest tine 10m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_z100m_InFirn_RG
--------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. Largest tine 100m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_z200m_InFirn_RG
---------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. Largest tine 200m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_InfAir
------------------------
Same as createLPDA_100MHz_InfFirn but antenna embedded in infinite air (i.e. n = 1).
Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z1cm_InAir_RG
--------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
lowest/largest tine 1cm above firn (this is because in the simulation the geometry is inverted, the ground is air and the medium of the antenna is firn).
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

dip7cm_hpol_infFirn
-------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA.
Horizontally orientated dipole antenna in infinite firn media(n=1.3 assumed).
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_hpol_z2m_InFirn_RG
-------------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 2m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_InfFirn
---------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. Vertically orientated dipole in infinite firn (n=1.3 assumed).
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [-90,90] Phi range [0,360] Freq range [100,1000]MHz

Last updated: 2018

dip7cm_z260mm_InFirn_RG
------------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 260cm above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z1m_InFirn_RG
--------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 1m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z2m_InFirn_RG
--------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 2m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z3m_InFirn_RG_NearHorizontalHD
--------------------------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 3m in firn.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,0.5] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z5m_InFirn_RG
---------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 5m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z10m_InFirn_RG
---------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 10m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z20m_InFirn_RG
----------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA.
Dipole center 20m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z100m_InFirn_RG
----------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 100m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z200m_InFirn_RG
-----------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 200m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_InfAir
-------------
WIPL-D simulation of KU dipole 52cm long. This antenna is used by ARIANNA.
Vertically orientated dipole in infinite air (n=1).
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [=90,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z270mm_InAir
--------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 270cm above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z1m_InAir
----------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 1m above firn.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z1m_InAir_RG_NearHorizontalHD
-------------------------------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 1m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,1] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z1m_InAir_RG_NearHorizontalHD2
--------------------------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 1m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,0.5] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z2m_InAir
----------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 2m above firn.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z5m_InAir
-----------------

WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 5m above firn.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

ARA_quadslot_data+measurement_fit
----------------------------------

Best fit model of NEC2+XFDTD simulation and measurement of the ARA quad-slot antenna (Hpol).
The antenna was put down a hole in a cube of ice of 6 m length.
"Sensors" were put near the edge of the cube, but not too close, and the emitted electric field at that location was obtained and then the realized gain was calculated at n=1.78.
Theta range [0,90] Phi range [0,360] Freq range [83.3,1050]MHz

Last updated: 2019

ARA_bicone_data+measurement_fit
-------------------------------

Best fit model of NEC2+XFDTD simulation and measurement of the ARA Bicone antenna (Vpol).
Antenna was put down a hole in a cube of ice of 6 m length.
"Sensors" were put near the edge of the cube, but not too close, and the emitted electric field at that location was obtained and then the realized gain was calculated at n=1.78.
Theta range [0,90] Phi range [0,360] Freq range [83.3,1050]MHz

Last updated: 2019

RNOG_vpol_4inch_center_1.73
----------------------------
xF simulations for the RNOG Vpol in a 5.75 inch borehole with index of refraction of ice n=1.78.
The antenna is placed in the center (x, y) of the borehole. An extra cubic interpolation is performed in frequencies (5 MHz step).
Theta range [0, 90] Phi range [0, 360] Freq range [0, 4200]MHz

Last updated: 2020

RNOG_vpol_4inch_half_1.73
--------------------------
xF simulations for the RNOG Vpol in a 5.75 inch borehole with index of refraction of ice n=1.78.
The antenna is halfway displaced from the center towards phi = 0. An extra cubic interpolation is performed in frequencies (5 MHz step).
Theta range [0, 90] Phi range [0, 360] Freq range [0, 4200]MHz

Last updated: 2020

RNOG_vpol_4inch_wall_1.73
-------------------------
xF simulations for the RNOG Vpol in a 5.75 inch borehole with index of refraction of ice n=1.78.
The antenna placed against the wall towards phi = 0. An extra cubic interpolation is performed in frequencies (5 MHz step).
Theta range [0, 90] Phi range [0, 360] Freq range [0, 4200]MHz

Last updated: 2020

RNOG_quadslot_v1_1.74
----------------------
XFdtd simulations for the RNOG Hpol.
Simulations are done in air, frequencies are rescaled with n=1.74. An extra cubic interpolation is performed in frequencies (5 MHz step).
Theta range [-180, 180] Phi range [0, 360] Freq range [57, 574]MHz

Last updated: 2020

RNOG_quadslot_v2_1.74
----------------------
XFdtd simulations for the RNOG Hpol.
Simulations are done in air, frequencies are rescaled with n=1.74. An extra cubic interpolation is performed in frequencies (5 MHz step).
Theta range [-180, 180] Phi range [0, 360] Freq range [57, 574]MHz

Last updated: 2020

SKALA_InAir
--------------
Log-periodic antenna for SKA-low, called SKALA-2.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
Theta range [0, 90]; Phi range [0, 360]; Freq range [50, 350]MHz
For more information, see: https://ieeexplore.ieee.org/abstract/document/7297231/authors#authors
Last updated: 2021



RNOG_quadslot_v3_air_rescaled_to_n1.74
--------------------------------------
XFdtd simulations in for the RNO-G Hpol.
Simulations are done in air, frequencies are rescaled with n=1.74. 
Theta range [-180, 180] Phi range [0, 360] Freq range [57, 574]MHz

Last updated: 2020
