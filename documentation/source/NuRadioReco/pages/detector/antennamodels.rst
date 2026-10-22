Antenna Models
=========================================

All antenna models are stored on a central data server and are downloaded automatically on-demand
whenever the user requests the antenna model for the first time.

For developers:
If you add a new antenna model please add the sha1sum = hashlib.sha1() to ``NuRadioReco/detector/antenna_models_hash.json``
as ``[sha1sum, comment]`` and send one of the core maintainers the antenna model so that they can put it on our central server.
The comment (may be empty) is logged whenever the model is loaded: at error level if it contains "error",
at warning level if it contains "warning" (case-insensitive), otherwise at info level.
The comment field was added in version 3.2; entries with only the sha1sum (as a plain string) are still supported.

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
The model names are the unique identifiers that need to be specified as ``ant_type`` in the detector description.

.. warning::
    Before v3.2, the detector appended ``_InfAir`` or ``_InfFirn`` to ``ant_type`` depending on the antenna
    depth (``antenna_by_depth``). Since v3.2 this is no longer done: ``ant_type`` must be the full model name
    as listed below (e.g., ``createLPDA_100MHz_v2_InfFirn`` instead of ``createLPDA_100MHz``).

Inf means usually an infinite medium, firn has a refractive index n = 1.3-1.4. Since this is a bit imprecise
we changed the naming to the actual refractive index.
Naming conventions:

* ``_InfAir`` / ``_InfFirn``: the antenna is embedded in an *infinite* homogeneous medium (no boundary).
  "Firn" has a refractive index of n = 1.3-1.4; since this is imprecise, newer models state the actual refractive
  index instead (e.g., ``_inf_n1.4``).
* ``_z<d>_InAir`` / ``_z<d>_InFirn``: the antenna is *in* air/firn at a distance ``d`` from the air-firn boundary
  (the reference point on the antenna is given in the model descriptions), which is included in the simulation. Unlike the infinite-medium models, these models are documented with a theta
  range of [0, 90] deg only.
* The meaning of the additional suffixes ``_RG``, ``_NoRG``, ``_Backlobe`` and ``_BoresightToBoundary`` is
  currently not documented.

The models are grouped by antenna family. Unless stated otherwise, the Phi range is [0, 360] deg.
Outdated models are listed in a collapsed block at the end of their family section (click to expand);
they should only be used for reproducing old results and a warning is logged whenever one of them is loaded.

Neutrino detector antennas
==========================

ARA bicone: bicone_v8
---------------------

WIPL-D simulations of the ARA bicone antenna, which has been used by ARIANNA at the South Pole.
The antenna is embedded in an infinite medium. The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _bicone_v8_InfAir:

       ``bicone_v8_InfAir``
     - Infinite air (n = 1.0).
     - [-90, 90]
     - [5, 1000]
     - 2018
   * - .. _bicone_v8_InfFirn:

       ``bicone_v8_InfFirn``
     - Identical to ``bicone_v8_inf_n1.32`` (same file hash).
     - [-90, 90]
     - [5, 1000]
     - 2018
   * - .. _bicone_v8_inf_n1.32:

       ``bicone_v8_inf_n1.32``
     - n = 1.32
     - [-90, 90]
     - [5, 1000]
     - 2018
   * - .. _bicone_v8_inf_n1.4:

       ``bicone_v8_inf_n1.4``
     - n = 1.4
     - [-90, 90]
     - [5, 1000]
     - 2018
   * - .. _bicone_v8_inf_n1.78:

       ``bicone_v8_inf_n1.78``
     - n = 1.78
     - [-90, 90]
     - [5, 1000]
     - 2018

Create LPDA: createLPDA_100MHz
------------------------------

WIPL-D simulations of the 100 MHz LPDA from Create, used by ARIANNA. The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.
For the ``*_InFirn_RG`` models the simulation geometry is inverted, i.e., the medium of the antenna is firn and the ground is air
(hence "above air" in their descriptions).

.. warning::
    ``createLPDA_100MHz_InfFirn`` and ``createLPDA_100MHz_InfFirn_n1.4`` use the wrong sign for the eTheta component.
    This is corrected in the ``v2`` versions, which therefore supersede them.

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _createLPDA_100MHz_InfAir:

       ``createLPDA_100MHz_InfAir``
     - Infinite air (n = 1).
     - [-90, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_InfFirn:

       ``createLPDA_100MHz_InfFirn``
     - **Wrong sign of the eTheta component**, superseded by ``createLPDA_100MHz_v2_InfFirn``. Infinite medium, n = 1.3.
     - [-90, 90]
     - [5, 1000] (assumed)
     - 2018
   * - .. _createLPDA_100MHz_v2_InfFirn:

       ``createLPDA_100MHz_v2_InfFirn``
     - Infinite medium, n = 1.3.
     - [-90, 90]
     - [5, 1000] (assumed)
     - 2025
   * - .. _createLPDA_100MHz_InfFirn_n1.4:

       ``createLPDA_100MHz_InfFirn_n1.4``
     - **Wrong sign of the eTheta component**, superseded by ``createLPDA_100MHz_v2_InfFirn_n1.4``. Infinite firn, n = 1.4.
     - [-90, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_v2_InfFirn_n1.4:

       ``createLPDA_100MHz_v2_InfFirn_n1.4``
     - Infinite firn, n = 1.4.
     - [-90, 90]
     - [5, 1000]
     - 2025
   * - .. _createLPDA_100MHz_z1cm_InFirn_RG:

       ``createLPDA_100MHz_z1cm_InFirn_RG``
     - Lowest/largest tine 1 cm above air.
     - [0, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z1cm_InFirn_BoresightToBoundary:

       ``createLPDA_100MHz_z1cm_InFirn_BoresightToBoundary``
     - Smallest/highest tine 1 cm below air.
     - [0, 90]
     - [20, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z10cm_InFirn_RG:

       ``createLPDA_100MHz_z10cm_InFirn_RG``
     - Lowest/largest tine 10 cm above air.
     - [0, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z1m_InFirn_RG:

       ``createLPDA_100MHz_z1m_InFirn_RG``
     - Lowest/largest tine 1 m above air.
     - [0, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z1m_InFirn_RG_v2:

       ``createLPDA_100MHz_z1m_InFirn_RG_v2``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _createLPDA_100MHz_z2m_InFirn_RG:

       ``createLPDA_100MHz_z2m_InFirn_RG``
     - Largest tine 2 m above air.
     - [0, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z2m_InFirn_Backlobe_NoRG:

       ``createLPDA_100MHz_z2m_InFirn_Backlobe_NoRG``
     - Largest tine 2 m above air.
     - [0, 90]
     - [200, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z3m_InFirn_BoresightToBoundary:

       ``createLPDA_100MHz_z3m_InFirn_BoresightToBoundary``
     - Largest tine 3 m below air; nose 1.58 m below air.
     - [0, 90]
     - [20, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z3mAndLPDALen_InFirn_BoresightToBoundary:

       ``createLPDA_100MHz_z3mAndLPDALen_InFirn_BoresightToBoundary``
     - Nose 3.2 m below air.
     - [0, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z5m_InFirn_RG:

       ``createLPDA_100MHz_z5m_InFirn_RG``
     - Largest tine 5 m above air.
     - [0, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z10m_InFirn_RG:

       ``createLPDA_100MHz_z10m_InFirn_RG``
     - Largest tine 10 m above air.
     - [0, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z100m_InFirn_RG:

       ``createLPDA_100MHz_z100m_InFirn_RG``
     - Largest tine 100 m above air.
     - [0, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z200m_InFirn_RG:

       ``createLPDA_100MHz_z200m_InFirn_RG``
     - Largest tine 200 m above air.
     - [0, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z1cm_InAir_RG:

       ``createLPDA_100MHz_z1cm_InAir_RG``
     - Lowest/largest tine 1 cm above firn.
     - [0, 90]
     - [5, 1000]
     - 2018
   * - .. _createLPDA_100MHz_z3m_InAir_RG:

       ``createLPDA_100MHz_z3m_InAir_RG``
     - Lowest/largest tine 3 m above firn.
     - [0, 90]
     - [5, 1000]
     - 2018

KU dipole: dip7cm
-----------------

WIPL-D simulations of the 52 cm long KU dipole, used by ARIANNA. The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided.

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _dip7cm_z260mm_InFirn_RG:

       ``dip7cm_z260mm_InFirn_RG``
     - Dipole center 260 mm in firn.
     - [0, 90]
     - [20, 1000]
     - 2018
   * - .. _dip7cm_z2m_InFirn_RG:

       ``dip7cm_z2m_InFirn_RG``
     - Dipole center 2 m in firn.
     - [0, 90]
     - [20, 1000]
     - 2018
   * - .. _dip7cm_z5m_InFirn_RG:

       ``dip7cm_z5m_InFirn_RG``
     - Dipole center 5 m above air.
     - [0, 90]
     - [20, 1000]
     - 2018
   * - .. _dip7cm_z10m_InFirn_RG:

       ``dip7cm_z10m_InFirn_RG``
     - Dipole center 10 m above air.
     - [0, 90]
     - [20, 1000]
     - 2018
   * - .. _dip7cm_z100m_InFirn_RG:

       ``dip7cm_z100m_InFirn_RG``
     - Dipole center 100 m below surface.
     - [0, 90]
     - [20, 1000]
     - 2018
   * - .. _dip7cm_z200m_InFirn_RG:

       ``dip7cm_z200m_InFirn_RG``
     - Dipole center 200 m below surface.
     - [0, 90]
     - [20, 1000]
     - 2018
   * - .. _dip7cm_z270mm_InAir:

       ``dip7cm_z270mm_InAir``
     - Dipole center 270 mm deep, in infinite air (n = 1).
     - [0, 90]
     - [20, 1000]
     - 2018
   * - .. _dip7cm_z1m_InAir:

       ``dip7cm_z1m_InAir``
     - Dipole center 1 m above firn.
     - [0, 90]
     - [20, 1000]
     - 2018

RNO-G VPol: RNOG_vpol
---------------------

XFdtd simulations of the RNO-G VPol antenna in a borehole.

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _RNOG_vpol_v3_5inch_center_IGLU_n1.74:

       ``RNOG_vpol_v3_5inch_center_IGLU_n1.74``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_vpol_v4_5inch_center_n1.490:

       ``RNOG_vpol_v4_5inch_center_n1.490``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_vpol_v4_5inch_center_n1.584:

       ``RNOG_vpol_v4_5inch_center_n1.584``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_vpol_v4_5inch_center_n1.656:

       ``RNOG_vpol_v4_5inch_center_n1.656``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_vpol_v4_5inch_center_n1.711:

       ``RNOG_vpol_v4_5inch_center_n1.711``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_vpol_v4_5inch_center_n1.728:

       ``RNOG_vpol_v4_5inch_center_n1.728``
     - *No description yet.*
     - 
     - 
     - 

.. raw:: html

   <details>
   <summary><b>Show outdated models</b> (as of 2026-10-15)</summary>

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _RNOG_vpol_4inch_center_n1.73:

       ``RNOG_vpol_4inch_center_n1.73``
     - 5.75 inch borehole, n = 1.73, antenna in the center (x, y) of the borehole. Extra cubic interpolation in frequency (5 MHz step). **Negative group delays, which are unphysical.**
     - [0, 90]
     - [0, 4200]
     - 2020
   * - .. _RNOG_vpol_4inch_half_n1.73:

       ``RNOG_vpol_4inch_half_n1.73``
     - 5.75 inch borehole, n = 1.73, antenna halfway displaced from the center towards phi = 0. Extra cubic interpolation in frequency (5 MHz step). **Negative group delays, which are unphysical.**
     - [0, 90]
     - [0, 4200]
     - 2020
   * - .. _RNOG_vpol_4inch_wall_n1.73:

       ``RNOG_vpol_4inch_wall_n1.73``
     - 5.75 inch borehole, n = 1.73, antenna placed against the wall towards phi = 0. Extra cubic interpolation in frequency (5 MHz step). **Negative group delays, which are unphysical.**
     - [0, 90]
     - [0, 4200]
     - 2020
   * - .. _RNOG_vpol_v1_n1.4:

       ``RNOG_vpol_v1_n1.4``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_vpol_v1_n1.73:

       ``RNOG_vpol_v1_n1.73``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_vpol_v2_5inch_center_n1.75:

       ``RNOG_vpol_v2_5inch_center_n1.75``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_vpol_v3_5inch_center_n1.74:

       ``RNOG_vpol_v3_5inch_center_n1.74``
     - 11.2 inch diameter borehole, n = 1.74, antenna in the center (x, y) of the borehole. No power feed-through cable included. Simulated for Theta [0, 90] and Phi [0, 90] due to simulation size constraints and extended using symmetry.
     - [0, 180]
     - [0, 1000]
     - 2025

.. raw:: html

   </details>

RNO-G HPol: RNOG_hpol, RNOG_quadslot, trislot_RNOG
--------------------------------------------------

XFdtd simulations of the RNO-G HPol antenna.

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _RNOG_hpol_v4_8inch_center_n1.74:

       ``RNOG_hpol_v4_8inch_center_n1.74``
     - 11.2 inch diameter borehole, n = 1.74, antenna in the center (x, y) of the borehole. No power feed-through cable included. Simulated for Theta [0, 90] and Phi [0, 90] due to simulation size constraints and extended using symmetry.
     - [0, 180]
     - [0, 1000]
     - 2025
   * - .. _RNOG_hpol_v4_8inch_center_IGLU_n1.74:

       ``RNOG_hpol_v4_8inch_center_IGLU_n1.74``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_hpol_v5_8inch_center_n1.728:

       ``RNOG_hpol_v5_8inch_center_n1.728``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_quadslot_v2_rescaled_fineFreq:

       ``RNOG_quadslot_v2_rescaled_fineFreq``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _RNOG_quadslot_v3_air_rescaled_to_n1.74:

       ``RNOG_quadslot_v3_air_rescaled_to_n1.74``
     - Simulated in air, frequencies rescaled with n = 1.74.
     - [-180, 180]
     - [57, 574]
     - 2020

.. raw:: html

   <details>
   <summary><b>Show outdated models</b> (as of 2026-10-15)</summary>

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _RNOG_quadslot_v1_n1.74:

       ``RNOG_quadslot_v1_n1.74``
     - Simulated in air, frequencies rescaled with n = 1.74. Extra cubic interpolation in frequency (5 MHz step).
     - [-180, 180]
     - [57, 574]
     - 2020
   * - .. _RNOG_quadslot_v2_n1.74:

       ``RNOG_quadslot_v2_n1.74``
     - Simulated in air, frequencies rescaled with n = 1.74. Extra cubic interpolation in frequency (5 MHz step).
     - [-180, 180]
     - [57, 574]
     - 2020
   * - .. _trislot_RNOG:

       ``trislot_RNOG``
     - *No description yet.*
     - 
     - 
     - 

.. raw:: html

   </details>

ARA: XFDTD
----------

*No description yet.*

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _XFDTD_Hpol_150mmHole_n1.78:

       ``XFDTD_Hpol_150mmHole_n1.78``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _XFDTD_Vpol_CrossFeed_150mmHole_n1.78:

       ``XFDTD_Vpol_CrossFeed_150mmHole_n1.78``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _XFDTD_Vpol_CrossFeed_150mmHole_n1.78_InfFirn:

       ``XFDTD_Vpol_CrossFeed_150mmHole_n1.78_InfFirn``
     - *No description yet.*
     - 
     - 
     - 

Other
-----

*No description yet.*

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _fourslot_InfFirn:

       ``fourslot_InfFirn``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _greenland_vpol_InfFirn:

       ``greenland_vpol_InfFirn``
     - *No description yet.*
     - 
     - 
     - 

Air-shower array antennas
=========================

AERA: Butterfly and SmallBlackSpider
------------------------------------

Antennas of the AERA detector at the Pierre Auger Observatory. The models contain the low-noise amplifier (LNA),
which is integrated into the antenna terminals. If you use these models in a paper, please cite: http://dx.doi.org/10.1088/1748-0221/7/10/P10011 and http://dx.doi.org/10.1016/j.nima.2011.01.049

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _Butterfly_ground2:

       ``Butterfly_ground2``
     - The Butterfly antenna, based on simulations.
     - 
     - 
     - 2025 (added)
   * - .. _Butterfly_ground2_East:

       ``Butterfly_ground2_East``
     - The East arm of the Butterfly antenna. Based on an in-situ measurement campaign (PhD thesis: http://doi.org/10.18154/RWTH-2018-225398).
     - 
     - 
     - 2025 (added)
   * - .. _Butterfly_ground2_North:

       ``Butterfly_ground2_North``
     - The North arm of the Butterfly antenna. Based on an in-situ measurement campaign (PhD thesis: http://doi.org/10.18154/RWTH-2018-225398).
     - 
     - 
     - 2025 (added)
   * - .. _SmallBlackSpider_ground2_measured:

       ``SmallBlackSpider_ground2_measured``
     - The Small Black Spider LPDA, based on an in-situ measurement campaign (10.1088/1748-0221/12/10/T10005). Because of the LNA, the vector effective length is of the order of 5-10 m.
     - 
     - 
     - 2025 (added)

LOFAR: LOFAR_LBA
----------------

LOFAR low-band antenna. ``LOFAR_LBA_X`` and ``LOFAR_LBA_Y`` are used in ``NuRadioReco/detector/LOFAR/LOFAR.json``.

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _LOFAR_LBA:

       ``LOFAR_LBA``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _LOFAR_LBA_X:

       ``LOFAR_LBA_X``
     - *No description yet.*
     - 
     - 
     - 
   * - .. _LOFAR_LBA_Y:

       ``LOFAR_LBA_Y``
     - *No description yet.*
     - 
     - 
     - 

SKA-Low: SKALA
--------------

Log-periodic antennas for SKA-Low.

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _SKALA_v4_Xpol:

       ``SKALA_v4_Xpol``
     - Embedded element simulation (FEKO) of the SKALA4 antenna, east-west orientation (see ``NuRadioReco.detector.antennapattern.preprocess_FEKO_mat``).
     - 
     - [50, 350]
     - 
   * - .. _SKALA_v4_Ypol:

       ``SKALA_v4_Ypol``
     - Same as ``SKALA_v4_Xpol`` but north-south orientation.
     - 
     - [50, 350]
     - 

.. raw:: html

   <details>
   <summary><b>Show outdated models</b> (as of 2026-10-15)</summary>

.. list-table::
   :header-rows: 1
   :widths: 30 46 8 8 8

   * - Model
     - Description
     - Theta [deg]
     - Freq [MHz]
     - Updated
   * - .. _SKALA_InfFirn:

       ``SKALA_InfFirn``
     - SKALA-2. The complex (magnitude + phase) vector effective length of both polarization components (ePhi, eTheta) is provided. See https://ieeexplore.ieee.org/abstract/document/7297231
     - [0, 90]
     - [50, 350]
     - 2021

.. raw:: html

   </details>
