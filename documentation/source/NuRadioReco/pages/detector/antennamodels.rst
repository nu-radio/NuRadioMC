Antenna Models
=========================================

All antenna models are stored on a central data server and are downloaded automatically on-demand
whenever the user requests the antenna model for the first time.

For developers:
If you add a new antenna model please add the sha1sum = hashlib.sha1() to ``NuRadioReco/detector/antenna_models_hash.json``
and send one of the core maintainers the antenna model so that they can put it on our central server.

Implementation of Antenna Models
================================

For the antenna orientation and rotation, the conventions are described in :ref:`Properties of Detector Description <NuRadioReco/pages/detector/detector_database_fields:Properties of Detector Description>`

The antenna models are accessed in the `AntennaPattern <NuRadioReco.detector.antennapattern.AntennaPattern>`
class in the `NuRadioReco.detector.antennapattern <NuRadioReco.detector.antennapattern>` module.
Different software packages are used to simulate the antennas, internally, NuRadioReco converts the data to a common pickle format in which they are
stored.

The antenna pickle files contains 9 lists of the following data:

    - orientation theta: float
        orientation of the antenna, as a zenith angle (0deg is the zenith, 180deg is straight down)
    - orientation phi: float
        orientation of the antenna, as an azimuth angle (counting from East counterclockwise)
    - rotation theta: float
        rotation of the antenna
    - rotation phi: float
        rotation of the antenna
    - freqs: array of floats
        array of frequencies for which the realized vector effective length is provided
    - theta: array of floats
        zenith angles for the realized vector effective length with respect to the antenna
    - phi: array of floats
        azimuth angles for the realized vector effective length with respect to the antenna
    - H_phi: array of floats
        the complex realized vector effective length of the ePhi polarization component as described in (A.13) of the NuRadioReco paper `arxiv:1903.07023 <https://arxiv.org/abs/1903.07023>`__
    - H_theta: array of floats
        the complex realized vector effective length of the eTheta polarization component as described in (A.13) of the NuRadioReco paper `arxiv:1903.07023 <https://arxiv.org/abs/1903.07023>`__

The calculation of the vector effective length is described in Appendix A.1, A.2 and A.3 of the NuRadioReco paper `arxiv:1903.07023 <https://arxiv.org/abs/1903.07023>`__.


Overview of available Antenna Models
====================================

The following antenna models are available in NuRadioReco.
The headings are the unique identifiers of the antenna model that need to specified as  ``antenna_type`` in the detector description.

Inf means usually an infinite medium, firn has a refractive index n = 1.3-1.4. Since this is a bit imprecise
we changed the naming to the actual refractive index.

bicone_v8_InfAir
-----------------
WIPL-D simulation of ARA Bicone antenna.
This antenna has been used by ARIANNA at the South Pole.
The antenna is embedded in an infinite medium with an index of refraction of n = 1.0.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

bicone_v8_InfFirn
------------------

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
.. warning::
    This model has the wrong sign for the eTheta component;
    this has been corrected in the :ref:`createLPDA_100MHz_v2_InfFirn <NuRadioReco/pages/detector/antennamodels:createLPDA_100MHz_v2_InfFirn>` model,
    which therefore supersedes this version.

WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
The antenna is embedded in an infinite medium with an index of refraction of n = 1.3.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz assumed

Last updated: 2018

createLPDA_100MHz_v2_InfFirn
----------------------------
WIPL-D simulation of 100 MHz LPDA from create. The ``v2`` version differs
only by a relative minus sign in the ``eTheta`` component; in the older version
the wrong sign was used.
This antenna is used by ARIANNA.
The antenna is embedded in an infinite medium with an index of refraction of n = 1.3.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [-90,90] Phi range [0,360] Freq range [5,1000] MHz assumed

Last updated: 2025

createLPDA_100MHz_InfFirn_n1.4
------------------------------
.. warning::
    This model has the wrong sign for the eTheta component;
    this has been corrected in the :ref:`createLPDA_100MHz_v2_InfFirn_n1.4 <NuRadioReco/pages/detector/antennamodels:createLPDA_100MHz_v2_InfFirn_n1.4>` model,
    which therefore supersedes this version.

Same as createLPDA_100MHz_InfFirn but antenna embedded in infinite firn with index of n = 1.4.

Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_v2_InfFirn_n1.4
---------------------------------
Same as createLPDA_100MHz_v2_InfFirn but antenna embedded in infinite firn with index of n = 1.4.
The ``v2`` version differs only by a relative minus sign in the ``eTheta`` component;
in the older version the wrong sign was used.

Theta range [-90,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2025

createLPDA_100MHz_z1cm_InFirn_RG
---------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
lowest/largest tine 1cm above air (this is because in the simulation the geometry is inverted, the ground is air and the medium of the antenna is firn).
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z1cm_InFirn_BoresightToBoundary
--------------------------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
smallest/highest tine 1cm below air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

createLPDA_100MHz_z10cm_InFirn_RG
----------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
lowest/largest tine 10cm above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z1m_InFirn_RG
--------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
lowest/largest tine 1m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z2m_InFirn_RG
--------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. Largest tine 2m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z2m_InFirn_Backlobe_NoRG
-------------------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. Largest tine 2m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [200,1000]MHz

Last updated: 2018

createLPDA_100MHz_z3m_InAir_RG
-------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. lowest/largest tine 3m above firn.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z3m_InFirn_BoresightToBoundary
-------------------------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
Largest tine 3m below air; nose 1.58m below air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

createLPDA_100MHz_z3mAndLPDALen_InFirn_BoresightToBoundary
-----------------------------------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. Nose 3.2m below air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z5m_InFirn_RG
--------------------------------
WIPL-D simulation of 100 MHz LPDA from create. This antenna is used by ARIANNA.
Largest tine 5m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z10m_InFirn_RG
---------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA.
Largest tine 10m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z100m_InFirn_RG
----------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. Largest tine 100m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_z200m_InFirn_RG
----------------------------------
WIPL-D simulation of 100 MHz LPDA from create.
This antenna is used by ARIANNA. Largest tine 200m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [5,1000]MHz

Last updated: 2018

createLPDA_100MHz_InfAir
-------------------------
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

createLPDA_100MHz_z1m_InFirn_RG_v2
-----------------------------------

dip7cm_hpol_infFirn
--------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA.
Horizontally orientated dipole antenna in infinite firn media(n=1.3 assumed).
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z260mm_InFirn_RG
------------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 260cm in firn.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z1m_InFirn_RG
---------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 1m in firn.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z2m_InFirn_RG
---------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 2m in firn.
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
----------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 10m above air.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z200m_InFirn_RG
-----------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA.
Dipole center 200m below surface.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z100m_InFirn_RG
-----------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 100m below surface.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018


dip7cm_infAir_s12
------------------
WIPL-D simulation of KU dipole 52cm long. This antenna is used by ARIANNA.
Vertically orientated dipole in infinite air (n=1).
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [=90,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z270mm_InAir
--------------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 270mm deep, in infinite air (n=1).
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z1m_InAir
-----------------
WIPL-D simulation of KU dipole 52cm long.
This antenna is used by ARIANNA. dipole center 1m above firn.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0,90] Phi range [0,360] Freq range [20,1000]MHz

Last updated: 2018

dip7cm_z1m_InAir_RG_NearHorizontalHD
-------------------------------------
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
-----------------
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

RNOG_vpol_4inch_center_n1.73
-----------------------------
xF simulations for the RNOG Vpol in a 5.75 inch borehole with index of refraction of ice n=1.73.
The antenna is placed in the center (x, y) of the borehole. An extra cubic interpolation is performed in frequencies (5 MHz step).

Theta range [0, 90] Phi range [0, 360] Freq range [0, 4200]MHz

Last updated: 2020

RNOG_vpol_4inch_half_n1.73
---------------------------
xF simulations for the RNOG Vpol in a 5.75 inch borehole with index of refraction of ice n=1.73.
The antenna is halfway displaced from the center towards phi = 0. An extra cubic interpolation is performed in frequencies (5 MHz step).

Theta range [0, 90] Phi range [0, 360] Freq range [0, 4200]MHz

Last updated: 2020

RNOG_vpol_4inch_wall_n1.73
---------------------------
xF simulations for the RNOG Vpol in a 5.75 inch borehole with index of refraction of ice n=1.73.
The antenna placed against the wall towards phi = 0. An extra cubic interpolation is performed in frequencies (5 MHz step).

Theta range [0, 90] Phi range [0, 360] Freq range [0, 4200]MHz

Last updated: 2020

RNOG_vpol_v3_5inch_center_n1.74
-------------------------------
XFdtd simulations for the RNO-G VPol in an 11.2 inch diameter borehole with index of refraction of ice n=1.74. The antenna is placed in the center (x, y) of the borehole.

Theta range [0, 180] Phi range [0, 360] Freq range [0, 1000]MHz. No power feed-through cable included.

Note: Simulation ran with Theta range [0, 90] and Phi range [0, 90] due to simulation size constraints and was extended to range noted above using symmetry.

Last updated: 2025

RNOG_hpol_v4_8inch_center_n1.74
-------------------------------
XFdtd simulations for the RNO-G HPol in an 11.2 inch diameter borehole with index of refraction of ice n=1.74. The antenna is placed in the center (x, y) of the borehole.

Theta range [0, 180] Phi range [0, 360] Freq range [0, 1000]MHz. No power feed-through cable included.

Note: Simulation ran with Theta range [0, 90] and Phi range [0, 90] due to simulation size constraints and was extended to range noted above using symmetry.

Last updated: 2025

RNOG_quadslot_v1_n1.74
-----------------------
XFdtd simulations for the RNOG Hpol.
Simulations are done in air, frequencies are rescaled with n=1.74. An extra cubic interpolation is performed in frequencies (5 MHz step).

Theta range [-180, 180] Phi range [0, 360] Freq range [57, 574]MHz

Last updated: 2020

RNOG_quadslot_v2_n1.74
-----------------------
XFdtd simulations for the RNOG Hpol.
Simulations are done in air, frequencies are rescaled with n=1.74. An extra cubic interpolation is performed in frequencies (5 MHz step).

Theta range [-180, 180] Phi range [0, 360] Freq range [57, 574]MHz

Last updated: 2020

RNOG_quadslot_v2_rescaled_fineFreq
-----------------------------------

RNOG_quadslot_v3_air_rescaled_to_n1.74
---------------------------------------
XFdtd simulations in for the RNO-G Hpol.
Simulations are done in air, frequencies are rescaled with n=1.74.

Theta range [-180, 180] Phi range [0, 360] Freq range [57, 574]MHz

Last updated: 2020

SKALA_InfFirn
--------------
Log-periodic antenna for SKA-low, called SKALA-2.
The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

Theta range [0, 90]; Phi range [0, 360]; Freq range [50, 350]MHz

For more information, see: https://ieeexplore.ieee.org/abstract/document/7297231/authors#authors

Last updated: 2021

SmallBlackSpider_ground2_measured
---------------------------------
The Small Black Spider LPDA of the AERA detector at the Pierre Auger Observatory.
The antenna model is based on an extensive in-situ measurement campaign which is documented in this paper: 10.1088/1748-0221/12/10/T10005
Please note that the model contains the low-noise amplifier (LNA) which is integrated into the antenna terminals. Therefore, the 
vector effective length is in the order of 5-10m.
If you use this model in a paper, please cite: http://dx.doi.org/10.1088/1748-0221/7/10/P10011 and http://dx.doi.org/10.1016/j.nima.2011.01.049
Added: 2025

Butterfly_ground2
-----------------
The Butterfly antenna of the AERA detector at the Pierre Auger Observatory. The antenna model is based on simulations.
Please note that the model contains the low-noise amplifier (LNA) which is integrated into the antenna terminals. 
If you use this model in a paper, please cite: http://dx.doi.org/10.1088/1748-0221/7/10/P10011 and http://dx.doi.org/10.1016/j.nima.2011.01.049
Added: 2025

Butterfly_ground2_East
----------------------
The East arm of the Butterfly antenna of the AERA detector at the Pierre Auger Observatory.
The antenna model is based on an extensive in-situ measurement campaign which is documented in this PhD thesis: http://doi.org/10.18154/RWTH-2018-225398
Please note that the model contains the low-noise amplifier (LNA) which is integrated into the antenna terminals. 
If you use this model in a paper, please cite: http://dx.doi.org/10.1088/1748-0221/7/10/P10011 and http://dx.doi.org/10.1016/j.nima.2011.01.049
Added: 2025

Butterfly_ground2_North
-----------------------
The North arm of the Butterfly antenna of the AERA detector at the Pierre Auger Observatory.
The antenna model is based on an extensive in-situ measurement campaign which is documented in this PhD thesis: http://doi.org/10.18154/RWTH-2018-225398
Please note that the model contains the low-noise amplifier (LNA) which is integrated into the antenna terminals. 
If you use this model in a paper, please cite: http://dx.doi.org/10.1088/1748-0221/7/10/P10011 and http://dx.doi.org/10.1016/j.nima.2011.01.049
Added: 2025


Additional Models
==================

RNOG_vpol_v1_n1.4
------------------

RNOG_vpol_v1_n1.73
-------------------

fourslot_InfFirn
-----------------

greenland_vpol_InfFirn
-----------------------

trislot_RNOG
-------------

dipole_ARA_bicone_infinitefirn
-------------------------------

XFDTD_Hpol_150mmHole_n1.78
---------------------------

XFDTD_Vpol_CrossFeed_150mmHole_n1.78
-------------------------------------

XFDTD_Vpol_CrossFeed_150mmHole_n1.78_InfFirn
---------------------------------------------
