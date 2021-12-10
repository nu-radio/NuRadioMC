# NuRadioMC/NuRadioReco
A Monte Carlo simulation package for radio neutrino detectors and reconstruction framework for radio detectors of high-energy neutrinos and cosmic-rays

The documentation can be found at https://nu-radio.github.io/NuRadioMC/main.html
Please visit the wiki at https://github.com/nu-radio/NuRadioMC/wiki for additional documentation.

If you want to keep up to date, consider signing up to the following email lists:
 * user email list, will be used to announce new versions and major improvements etc. Subscribe via https://lists.uu.se/sympa/subscribe/physics-astro-nuradiomc
 * developer email list, will be used to discuss the future development of NuRadioMC/Reco. Subscribe via: https://lists.uu.se/sympa/subscribe/physics-astro-nuradiomc-dev

If you're using NuRadioMC for your research, please cite

* C. Glaser, D. Garcia-Fernandez, A. Nelles et al., "NuRadioMC: Simulating the radio emission of neutrinos from interaction to detector", [European Physics Journal C 80, 77 (2020)](https://dx.doi.org/10.1140/epjc/s10052-020-7612-8), [arXiv:1906.01670](https://arxiv.org/abs/1906.01670)

and for the detector simulation and event reconstruction part

* C. Glaser, A. Nelles, I. Plaisier, C. Welling et al., "NuRadioReco: A reconstruction framework for radio neutrino detectors", [Eur. Phys. J. C (2019) 79: 464](https://dx.doi.org/10.1140/epjc/s10052-019-6971-5), [arXiv:1903.07023](https://arxiv.org/abs/1903.07023)



NuRadioMC is continuously improved and new features are being added. The following papers document new features (in reverse chronological order):


* B. Oeyen, I. Plaisier, A. Nelles, C. Glaser, T. Winchen, "Effects of firn ice models on radio neutrino simulations using a RadioPropa ray tracer", [PoS(ICRC2021)1027](https://doi.org/10.22323/1.395.1027)  (adds numerical ray tracer RadioPropa to allow signal propagation in arbitrary 3D index-of-refraction profiles)

* C. Glaser D. García-Fernández and A. Nelles, "Prospects for neutrino-flavor physics with in-ice radio detectors", [PoS(ICRC2021)1231](https://doi.org/10.22323/1.395.1231) (generalizes NuRadioMC to simulate the radio emission from any number of in-ice showers including their interference)

* D. García-Fernández, C. Glaser and A. Nelles, “The signatures of secondary leptons in radio-neutrino detectors in ice”, [Phys. Rev. D 102, 083011](https://dx.doi.org/10.1103/PhysRevD.102.083011), [arXiv:2003.13442](https://arxiv.org/abs/2003.13442) (addition of secondary interactions of muons and taus) 


If you would like to contribute, please contact @cg-laser or @anelles for permissions to work on NuRadioMC. We work with pull requests only that can be merged after review.
Also please visit https://github.com/nu-radio/NuRadioMC/blob/master/CONTRIBUTING.md

NuRadioMC is used in an increasing number of studies. To get an overview for what NuRadioMC can be used for, please have a look at the following publications or see [here](https://inspirehep.net/literature?sort=mostrecent&size=25&page=1&q=refersto%3Arecid%3A1738571%20or%20refersto%3Arecid%3A1725583):


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
* RNO-G collaboration, "Reconstructing the neutrino energy for in-ice radio detectors : A study for the Radio Neutrino Observatory Greenland (RNO-G)", [arXiv:2107.02604](https://arxiv.org/abs/2107.02604)
* Ice-Cube-Gen2 collaboration, "IceCube-Gen2: The Window to the Extreme Universe", [J.Phys.G 48 (2021) 6, 060501](https://doi.org/10.1088/1361-6471/abbd48), [arXiv:2008.04323](https://arxiv.org/abs/2008.04323)
* C. Welling et al., "Reconstructing non-repeating radio pulses with Information Field Theory", [JCAP 04 (2021) 071](https://doi.org/10.1088/1475-7516/2021/04/071), [arXiv:2102.00258](https://arxiv.org/abs/2102.00258)
* C. Glaser, S. Barwick, "An improved trigger for Askaryan radio detectors", [JINST 16 (2021) 05, T05001](https://doi.org/10.1088/1748-0221/16/05/T05001), [arXiv:2011.12997](https://arxiv.org/abs/2011.12997)
* RNO-G collaboration, "Design and Sensitivity of the Radio Neutrino Observatory in Greenland (RNO-G)", [JINST 16 (2021) 03, P03025](https://doi.org/10.1088/1748-0221/16/03/P03025) [arXiv:2010.12279](https://arxiv.org/abs/2010.12279)
* ARIANNA collaboration, "Probing the angular and polarization reconstruction of the ARIANNA detector at the South Pole", [JINST 15 (2020) 09, P09039](https://doi.org/10.1088/1748-0221/15/09/P09039), [arXiv:2006.03027](https://arxiv.org/abs/2006.03027)
