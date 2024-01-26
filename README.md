# NuRadioMC and NuRadioReco
NuRadioMC: A Monte Carlo simulation package for radio neutrino detectors.

NuRadioReco: A reconstruction and detector simulation framework for radio detectors of high-energy neutrinos and cosmic-rays. (NuRadioReco is independet of NuRadioMC 
and used by NuRadioMC for the detector and trigger simulation. To simplify the development and continuous integration testing, both codes are developed 
in the same github repository.)

The documentation can be found at https://nu-radio.github.io/NuRadioMC/main.html


If you're using NuRadioMC for your research, please cite

* C. Glaser, D. Garcia-Fernandez, A. Nelles et al., "NuRadioMC: Simulating the radio emission of neutrinos from interaction to detector", [European Physics Journal C 80, 77 (2020)](https://dx.doi.org/10.1140/epjc/s10052-020-7612-8), [arXiv:1906.01670](https://arxiv.org/abs/1906.01670)

and for the detector simulation and event reconstruction part

* C. Glaser, A. Nelles, I. Plaisier, C. Welling et al., "NuRadioReco: A reconstruction framework for radio neutrino detectors", [Eur. Phys. J. C (2019) 79: 464](https://dx.doi.org/10.1140/epjc/s10052-019-6971-5), [arXiv:1903.07023](https://arxiv.org/abs/1903.07023)


NuRadioMC is continuously improved and new features are being added. The following papers document new features (in reverse chronological order):

* N. Heyer and C. Glaser, “First-principle calculation of birefringence effects for in-ice radio detection of neutrinos”, [arXiv:2205.06169](https://arxiv.org/abs/2205.15872)  (adds birefringence modelling to NuRadioMC)

* B. Oeyen, I. Plaisier, A. Nelles, C. Glaser, T. Winchen, "Effects of firn ice models on radio neutrino simulations using a RadioPropa ray tracer", [PoS(ICRC2021)1027](https://doi.org/10.22323/1.395.1027)  (adds numerical ray tracer RadioPropa to allow signal propagation in arbitrary 3D index-of-refraction profiles)

* C. Glaser D. García-Fernández and A. Nelles, "Prospects for neutrino-flavor physics with in-ice radio detectors", [PoS(ICRC2021)1231](https://doi.org/10.22323/1.395.1231) (generalizes NuRadioMC to simulate the radio emission from any number of in-ice showers including their interference)

* D. García-Fernández, C. Glaser and A. Nelles, “The signatures of secondary leptons in radio-neutrino detectors in ice”, [Phys. Rev. D 102, 083011](https://dx.doi.org/10.1103/PhysRevD.102.083011), [arXiv:2003.13442](https://arxiv.org/abs/2003.13442) (addition of secondary interactions of muons and taus) 


If you would like to contribute, please contact @cg-laser or @anelles for permissions to work on NuRadioMC. We work with pull requests only that can be merged after review.
Also please visit https://nu-radio.github.io/NuRadioMC/Introduction/pages/contributing.html for details on our workflow and coding conventions.


## Publications builing up on NuRadioMC/Reco
NuRadioMC is used in an increasing number of studies. To get an overview for what NuRadioMC can be used for, please have a look at the following publications or see [here](https://inspirehep.net/literature?sort=mostrecent&size=25&page=1&q=refersto%3Arecid%3A1738571%20or%20refersto%3Arecid%3A1725583):

* V. Valera, M. Bustamante, O. Mena, "Joint measurement of the ultra-high-energy neutrino spectrum and cross section", [arXiv:2308.07709](https://arxiv.org/abs/2308.07709)
* IceCube-Gen2 collaboration, [IceCube-Gen2 Technical Design Report](https://icecube-gen2.wisc.edu/science/publications/TDR)
* ARIANNA collaboration (A. Anker et al.), "Developing New Analysis Tools for Near Surface Radio-based Neutrino Detectors", [arXiv:2307.07188](https://arxiv.org/abs/2307.07188)
* L. Pyras, C. Glaser S. Hallmann and A. Nelles, "Atmospheric muons at PeV energies in radio neutrino detectors", JCAP 10 (2023) 043, [arXiv:2307.04736](https://arxiv.org/abs/2307.04736)
* I. Plaisier, S. Bouma, A. Nelles, "Reconstructing the arrival direction of neutrinos in deep in-ice radio detectors", [arXiv:2302.00054](https://arxiv.org/abs/2302.00054)
* S. Bouma, A. Nelles for the IceCube-Gen2 collaboration, "Direction reconstruction performance for IceCube-Gen2 Radio", [PoS(ICRC2023)1045](https://pos.sissa.it/444/1045/pdf)
* F. Schlüter and S. Toscano for the IceCube-Gen2 collaboration, "Estimating the coincidence rate between the optical and radio array of IceCube-Gen2", [PoS(ICRC2023)1022](https://pos.sissa.it/444/1022/pdf)
* C. Glaser, A. Coleman and T. Glusenkamp, "NuRadioOpt: Optimization of Radio Detectors of Ultra-High Energy Neutrinos through Deep Learning and Differential Programming", [PoS(ICRC2023)1114](https://pos.sissa.it/444/1114/pdf) 
* A. Coleman and C. Glaser for the RNO-G collaboration, "Enhancing the Sensitivity of RNO-G Using a Machine-learning Based Trigger", [PoS(ICRC2023)1100](https://pos.sissa.it/444/1100/pdf)
* N. Heyer, C. Glaser and T. Glusenkamp for the IceCube-Gen2 collaboration, "Deep Learning Based Event Reconstruction for the IceCube-Gen2 Radio Detector" [PoS(ICRC2023)1102](https://pos.sissa.it/444/1102/pdf)
* N. Heyer and C. Glaser, "Impact of Birefringence on In-Ice Radio Detectors of ultra-high-energy Neutrinos", [PoS(ICRC2023)1101](https://pos.sissa.it/444/1101/pdf)
* J. Henrichs, A. Nelles for the RNO-G Collaboration, "Searching for cosmic-ray air showers with RNO-G", [PoS(ICRC2023)259](https://pos.sissa.it/444/259/pdf)
* B. Oeyen for the RNO-G Collaboration, "The interplay of ice-firn model and station calibration in RNO-G", [PoS(ICRC2023)1042](https://pos.sissa.it/444/1042/pdf)
* P. Windischhofer, C. Welling and C. Deaconu, "Eisvogel: Exact and efficient calculations of radio emissions from in-ice neutrino showers", [PoS(ICRC2023)1157](https://pos.sissa.it/444/1157/)
* V. B. Valera, M. Bustamante and C. Glaser, “Near-future discovery of the diffuse flux of ultra-high-energy cosmic neutrinos”, Phys. Rev. D 107, 043019 [arXiv:2210.03756](https://arxiv.org/abs/2210.03756)
* Alfonso Garcia Soto, Diksha Garg, Mary Hall Reno, Carlos A. Argüelles, "Probing Quantum Gravity with Elastic Interactions of Ultra-High-Energy Neutrinos", Phys. Rev. D 107, 033009 (2023) [arXiv:2209.06282](https://arxiv.org/abs/2209.06282)
* Damiano F. G. Fiorillo, Mauricio Bustamante, Victor B. Valera, "Near-future discovery of point sources of ultra-high-energy neutrinos", JCAP 03 (2023) 026 [arXiv:2205.15985](https://arxiv.org/abs/2205.15985)
* C. Glaser, S. McAleer, S. Stjärnholm, P. Baldi, S. W. Barwick, “Deep learning reconstruction of the neutrino direction and energy from in-ice radio detector data”, Astroparticle Physics 145, (2023) 102781, [doi:10.1016/j.astropartphys.2022.102781](https://doi.org/10.1016/j.astropartphys.2022.102781), [arXiv:2205.15872](https://arxiv.org/abs/2205.15872)
* J. Beise and C. Glaser, “In-situ calibration system for the measurement of the snow accumulation and the index-of-refraction profile for radio neutrino detectors”, Journal of Instrumentation 18 P01036 (2023), [arXiv:2205.00726](https://arxiv.org/abs/2205.00726)
* V. B. Valera, M. Bustamante and C. Glaser, “The ultra-high-energy neutrino-nucleon cross section: measurement forecasts for an era of cosmic EeV-neutrino discovery”, Journal of High Energy Physics 06 (2022) 105, [doi:10.1007/JHEP06(2022)105](https://doi.org/10.1007/JHEP06(2022%29105), [arXiv:2204.04237](https://arxiv.org/abs/2204.04237)
* ARIANNA collaboration (A. Anker et al.), “Measuring the Polarization Reconstruction Resolution of the ARIANNA Neutrino Detector with Cosmic Rays”, Journal of Cosmology and Astroparticle Physics 04(2022)022, [doi:10.1088/1475-7516/2022/04/022](https://doi.org/10.1088/1475-7516/2022/04/022), [arXiv:2112.01501](https://arxiv.org/abs/2112.01501)
* ARIANNA collaboration (A. Anker et al.), “Improving sensitivity of the ARIANNA detector by rejecting thermal noise with deep learning”, Journal of Instrumentation 17 P03007 (2022), [doi:10.1088/1748-0221/17/03/P03007](https://doi.org/10.1088/1748-0221/17/03/P03007), [arXiv:2112.01031](https://arxiv.org/abs/2112.01031)
* RNO-G collaboration (J. A. Aguilar et al.), “Reconstructing the neutrino energy for in-ice radio detectors”, European Physics Journal C (2022) 82:147, [doi:10.1140/epjc/s10052-022-10034-4](https://doi.org/10.1140/epjc/s10052-022-10034-4), [arXiv:2107.02604](https://arxiv.org/abs/2107.02604)
* S. Stjärnholm, O. Ericsson and C. Glaser, "Neutrino direction and flavor reconstruction from radio detector data using deep convolutional neural networks", [PoS(ICRC2021)1055](https://doi.org/10.22323/1.395.1055)
* S. Hallmann et al., "Sensitivity studies for the IceCube-Gen2 radio array", [PoS(ICRC2021)1183](https://doi.org/10.22323/1.395.1183)
* Y. Pan, "A neural network based UHE neutrino reconstruction method for the Askaryan Radio Array (ARA)", [PoS(ICRC2021)1157](https://doi.org/10.22323/1.395.1157)
* A. Anker et al., "A novel trigger based on neural networks for radio neutrino detectors", [PoS(ICRC2021)1074](https://doi.org/10.22323/1.395.1074)
* L. Zhao et al., "Polarization Reconstruction of Cosmic Rays with the ARIANNA Neutrino Radio Detector", [PoS(ICRC2021)1156](https://doi.org/10.22323/1.395.1156)
* J. Beise et al. "Development of an in-situ calibration device of firn properties for Askaryan neutrino detectors", [PoS(ICRC2021)1069](https://doi.org/10.22323/1.395.1069)
* I. Plaisier et al., "Direction reconstruction for the Radio Neutrino Observatory Greenland", [PoS(ICRC2021)1026](https://doi.org/10.22323/1.395.1026)
* C. Welling et al., "Energy reconstruction with the Radio Neutrino Observatory Greenland (RNO-G)", [PoS(ICRC2021)1033](https://doi.org/10.22323/1.395.1033)
* C. Glaser, S. McAleer, P. Baldi and S.W. Barwick, "Deep learning reconstruction of the neutrino energy with a shallow Askaryan detector", [PoS(ICRC2021)1051](https://doi.org/10.22323/1.395.1051)
* S. Barwick et al., "Capabilities of ARIANNA: Neutrino Pointing Resolution and Implications for Future Ultra-high Energy Neutrino Astronomy", [PoS(ICRC2021)1151](https://doi.org/10.22323/1.395.1151)
* S. Barwick et al., "Science case and detector concept for ARIANNA high energy neutrino telescope at Moore's Bay, Antarctica", [PoS(ICRC2021)1190](https://doi.org/10.22323/1.395.1190)
* Ice-Cube-Gen2 collaboration, "IceCube-Gen2: The Window to the Extreme Universe", [J.Phys.G 48 (2021) 6, 060501](https://doi.org/10.1088/1361-6471/abbd48), [arXiv:2008.04323](https://arxiv.org/abs/2008.04323)
* C. Welling et al., "Reconstructing non-repeating radio pulses with Information Field Theory", [JCAP 04 (2021) 071](https://doi.org/10.1088/1475-7516/2021/04/071), [arXiv:2102.00258](https://arxiv.org/abs/2102.00258)
* C. Glaser, S. Barwick, "An improved trigger for Askaryan radio detectors", [JINST 16 (2021) 05, T05001](https://doi.org/10.1088/1748-0221/16/05/T05001), [arXiv:2011.12997](https://arxiv.org/abs/2011.12997)
* RNO-G collaboration, "Design and Sensitivity of the Radio Neutrino Observatory in Greenland (RNO-G)", [JINST 16 (2021) 03, P03025](https://doi.org/10.1088/1748-0221/16/03/P03025) [arXiv:2010.12279](https://arxiv.org/abs/2010.12279)
* ARIANNA collaboration, "Probing the angular and polarization reconstruction of the ARIANNA detector at the South Pole", [JINST 15 (2020) 09, P09039](https://doi.org/10.1088/1748-0221/15/09/P09039), [arXiv:2006.03027](https://arxiv.org/abs/2006.03027)


If you want to keep up to date, consider signing up to the following email lists:
 * user email list, will be used to announce new versions and major improvements etc. Subscribe via https://lists.uu.se/sympa/subscribe/physics-astro-nuradiomc
 * developer email list, will be used to discuss the future development of NuRadioMC/Reco. Subscribe via: https://lists.uu.se/sympa/subscribe/physics-astro-nuradiomc-dev
