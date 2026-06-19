Likelihood-based forward folding reconstruction
===============================================

A likelihood-based approach to forward-folding reconstruction was recently presented
in https://arxiv.org/abs/2510.21925. This approach uses a correct probablilistic description
of band-limited noise and parameterized signal models to achieve the most accurate reconstruction
resolution and correctly estimate uncertainties on the reconstructed parameters. This method is
applicable to, e.g., neutrino reconstruction and cosmic-ray electric-field reconstruction. The
main modules used to make the results in the paper are implemented in NuRadioReco and presented
here.

The code consists of 4 core modules that handle different parts of the reconstruction:
  - `NuRadioReco.modules.likelihood_reconstruction.likelihood_calculator.LikelihoodCalculator`: Calculates the likelihood
  for a simulated signal trace given a measured data trace. This module must be initialized with the
  spectra of the noise in the traces, or using many traces consisting purely of noise.
  - `NuRadioReco.modules.likelihood_reconstruction.shower_simulator.ShowerSimulator`: Simulates the signal traces from a
  neutrino induced shower for user-defined shower parameters, a given detector, and a simulation config
  file. This acts as a pure neutrino signal model and is used in forward-folding reconstruction of neutrino
  signals.
  - `NuRadioReco.utilities.minimization.Minimization`: Class for minimization of, e.g., a -2 log likelihood that unifies
  the interfaces of `scipy.optimize` and `iminuit`. The class adds additional functionality like normalization
  of the fitted parameters.
  - `NuRadioReco.utilities.matched_filter.MatchedFilter`: Class for performing a mathced filter search of simualated signal
  templates in noisy data. The class is initialized with the spectra of the noise in each antenna. This is
  a useful method to identify signal traces from backgrounds in an event-selection. It is also used in reconstrution
  algorithms to efficiently profile over time and amplitude and improve reconstruction stability.

An example script demonstrating how to use the core modules is available in `NuRadioReco/examples/likelihood_reconstruction/run_shower_simulator.py` and a script demonstrating the `ShowerSimulator` is available here `/NuRadioReco/examples/likelihood_reconstruction/toy_signal_likelihood_fit_and_matched_filter.py`.


These modules are then combined in modules for specific reconstruction tasks with easy to use interfaces:
  - `NuRadioReco.modules.likelihood_reconstruction.neutrinoLikelihoodReconstructor`: Reconstructs a neutrino
  signal using the data traces in a `station` object, the spectra of the noise for each antenna, a detector
  description, and a simualtion config file.
  - `NuRadioReco.modules.likelihood_reconstruction.electricFieldLikelihoodReconstructor`: Reconstructs a pulsed
  electric field in an ensemble of antennas in close proximity to each other using the data traces in a `station`
  object. The method assumes that all antennas observe the same electric field and uses an analytic parametrization
  of an electric field which is forward-folded through the antenna responses. This can be used to reconstruct
  cosmic-ray electric-fields in dual-polarized antennas or in RNO-G shallow stations.

Neutrino reconstruction
-----------------------

The `neutrinoLikelihoodReconstructor` reconstructs the parameters of a shower from a neutrino signal given a set of measured traces.
The code assumes that you have `station` object with the measured traces stored in the channels, that the noise spectra are
known, and a detector description (`det`) corresponding to the data. The reconstructor is then intialized with using a
user-defined simulation config file and `detector_simulation_filter_amp`:

.. code-block:: Python

    import NuRadioReco.modules.likelihood_reconstruction.neutrinoLikelihoodReconstructor
    reco = NuRadioReco.modules.likelihood_reconstruction.neutrinoLikelihoodReconstructor.neutrinoLikelihoodReconstructor()
    reco.begin(
        n_channels = det.get_number_of_channels(station_id),
        n_samples = det.get_number_of_samples(station_id, 0),
        sampling_rate = det.get_sampling_frequency(station_id, 0),
        noise_spectra = np.abs(filt),
        Vrms = noise_amplitude,
        detector_simulation_filter_amp = detector_simulation_filter_amp,
    )

To run the reconstruction, a good guess of the initial parameters is needed, which can be obtained from other reconstruction
methdod like interferometry-based vertex reconstruction. The convention used for the parametrization can be seen in the
documentation of the class. The reconstruction is then run with:

.. code-block:: Python

    reco.run(
        evt,
        station,
        det,
        parameters_initial,
        reference_channel = 0,
        full_output = False
    )

which saves the reconstructed shower as a `shower` in the `station` object. Alternatively, `full_output = True`, the reconstructed parameters, signal, likelihood values, and fit p-value are returned.

A full example of how to run the `neutrinoLikelihoodReconstructor` is shown in `NuRadioReco/examples/likelihood_reconstruction/neutrino_signal_reconstruction.py` 


Electric-field reconstruction
-----------------------------

The `electricFieldLikelihoodReconstructor` assumes that the data traces are stored in a station object. The reconstruction module is then intialized with:

.. code-block:: Python

    import NuRadioReco.modules.likelihood_reconstruction.electricFieldLikelihoodReconstructor
    reco = NuRadioReco.modules.likelihood_reconstruction.electricFieldLikelihoodReconstructor.electricFieldLikelihoodReconstructor()
    reco.begin(
        n_channels = det.get_number_of_channels(station_id),
        n_samples = det.get_number_of_samples(station_id, 0),
        sampling_rate = det.get_sampling_frequency(station_id, 0),
        noise_spectra = np.abs(filt),
        Vrms = noise_amplitude,
        filter_settings_list =  [filter_settings_low, filter_settings_high],
    )

If ray-traced travel times are needed, they can be provided through `travel_time_shifts`. 

.. code-block:: Python

    reco.run(
        evt,
        station,
        det,
        use_MC_direction = False
        full_output = False
    )


If is False, the already reconstructed arrival direction stored in `station[stnp.zenith]` and `station[stnp.azimuth]` are used. The reconstructed electric field are saved as an `electric_field` in the `station` object. Alternatively, `full_output = True`, the reconstructed, signal, signal parameters, and fitted likelihood value are returned.

A full example of how to run the `electricFieldLikelihoodReconstructor` is shown in `NuRadioReco/examples/likelihood_reconstruction/electric_field_reconstruction.py` 




