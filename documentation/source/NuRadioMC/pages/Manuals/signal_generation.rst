Signal Generation (electric field)
====================================
One of the NuRadioMC pillars is the signal generation procedure. All the relevant modules for signal generation can be found in the SignalGen folder. NuRadioMC has two methods for generating the electric field. The first one is through the use of frequency-domain parameterisations, and the second one is the use of the semianalytical ARZ model. An more in-depth explanation of the models can be found in the `NuRadioMC paper <https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-020-7612-8>`__.

Frequency-domain parameterisations
-----------------------------------
Frequency-domain parameterisations are fits to simulation results for which the fitting functions are motivated by physical arguments. These parameterisations return waveforms and spectra that closely resemble the results of of full Monte Carlos. For instance, their predicted amplitudes are not far off of what a refined Monte Carlo would yield, so they can be used for simple trigger studies and for effective volume calculations. However, the fine details of the shower emission, an accurate timing, shower-to-shower fluctuations, or LPM effect cannot be fully reproduced with these frequency domain parameterisations. These parameterisations are:

    * ``ZHS1992``: The first parameterisation of its kind, from the seminal `ZHS paper <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.45.362>`__. Included mainly for historical reasons.
    * ``Alvarez2000``: A more accurate version of the ZHS1992 parameterisation. One of the most used by the radio neutrino community. More information can be found in `this paper <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.62.063001>`__.
    * ``Alvarez2009``: The most complete of the parameterisations available in NuRadioMC. It possesses two different parameterisations: one for hadronic and another one for electromagnetic cascades. It also has a stochastic method (not incredibly accurate for individual pulses, but it reproduces effective volumes quite well) for emulating the LPM effect in electromagnetic cascades at high energies. More information can be found in `this paper <http://www.sciencedirect.com/science/article/pii/S0927650509001029>`__.

We vividly recommend the use of the Alvarez2009 parameterisation, which is the one that comes by default, for most of the applications. If a higher accuracy is needed, the user should resort to the ARZ models.

We reproduce here a simple example of how the parameterisations can be used:

    .. code-block:: Python

        from NuRadioReco.utilities import units
        from NuRadioMC.SignalGen.askaryan import get_time_trace

        energy = 1 * units.EeV
        theta = 57 * units.deg # Viewing angle w.r.t. shower axis
        N = 2048 # Number of samples 
        dt = 0.2 * units.ns # Time binning
        shower_type = 'HAD' # 'HAD' or 'EM'
        R = 1 * units.km # Distance to shower vertex
        n_index = 1.78 # refractive index

        model = 'Alvarez2009'

        trace = get_time_trace(energy, theta, N, dt, shower_type, 
                               n_index, R, model)

The trace returned by ``get_time_trace`` is a one-dimensional array. This is enough to characterise the electric field because, since it is a radiation field from a shower, it is polarised on the direction perpendicular to the line of sight between the observer and the emitter. For the parameterisations, it is assumed that the shower dimensions are negligible with respect to the distance between shower and observer. The ``theta`` parameter controls the angle formed between the shower axis and the vertex-observer line.

The trace units are the standard units for an electric field in NuRadioMC, that is, volts per metre. The user can also obtain the Fourier transform using the function ``get_frequency_spectrum``. An FFT wrapper coded in NuRadioReco ensures that the units of this spectrum are units of electric field per frequency (volts per metre per gigahertz, in default NuRadioMC units), rejecting the standard adimensional FFT convention and freeing the user of keeping track of the transform normalisation.

ARZ - semi-analytical model
-----------------------------
NuRadioMC posesses an implementation of the semi-analytical `ARZ model <https://dx.doi.org/10.1103/PhysRevD.84.103003>`__. This model uses a library of simulated shower longitudinal profiles together with a simulated parameterisation of the electric field at the Cherenkov angle and a formula for calculating the electric field for every observer using these data. This method agrees with the ZHS MC to the few percent level up to ~3 GHz. This model offers the advantage of having shower-to-shower fluctuations thanks to the simulated shower profiles library.  These showers have been simulated using a model for the LPM effect, which means that high-energy electromagnetic showers will be stretched and can present several local shower maxima. Also, hadronic showers can have a displaced maximum due to a pion that takes a lot of time to decay. All of the shower physics complexity is present if the library contains a fair sample of neutrino-induced showers, which translates into an accurate depiction of the electric field.

When the user calls the ARZ module, the shower library ``ARZ/shower_profile/shower_library_vX.Y.pkl`` will be loaded in memory, and the shower profiles will be drawn from there. However, advanced users can provide their own shower profiles using a format similar to the files ``ARZ/shower_profile/nue*tXXXX`` and then use the file ``A01preprocess_shower_library.py`` to create the pickle library.

The ARZ module should be the one used for reconstruction purposes, while the Alvarez2009 module can be used for effective volume calculations. However, if time is not an issue, we also recommend the use of the ARZ module for effective volume calculations.

The disadvantage of the ARZ module is that a numerical integration must be performed, so it is slower than the frequency-domain parameterisations. The way NuRadioMC integrates this formula by default is by using the trapezoid rule in two different regions. Where the integrand is larger and therefore results in more electric field, a fine subdivision is used, as opposed to the rest of the integration region, where the subdivisions are larger. This is controlled by the parameters ``interp_factor`` (for the less contributing region, default 1) and ``interp_factor2`` (for the most contributing region, default 100).

To calculate the electric field with the ARZ module, it suffices to use the example above with:

    .. code-block:: Python

        model = 'ARZ2020' # or 'ARZ2019'

The ARZ2019 model is a hadronic extension of the ARZ model. The ARZ2020 model is an update of ARZ2019 with slightly better fits (`ARZ2020 paper <https://dx.doi.org/10.1103/PhysRevD.101.083005>`__).

The trace returned by ``get_time_trace`` in this case is also a one-dimensional array containing the projection of the field perpendicular to the line of sight. Due to the extension of the shower in this model, there is a small electric field component  that parallel to the line of sight. However, this field is ignored by the function, as it is rather small. If this radial field is big, that's an indication that the observer is too close to the shower and the ARZ method is not valid anyway.

While the function ``get_time_trace`` in askaryan.py can be used for our simulations, if the user wishes to study the output of the ARZ model to better know the electric field, it is recommended to use the ``SignalGen/ARZ/ARZ.py`` module. This gives access to the three-dimensional electric field, and it also allows the user to specify the distance and viewing angle with respect to either the shower maximum or the shower vertex.

This piece of code illustrates how to use the ARZ module directly.

    .. code-block:: Python

        from NuRadioMC.SignalGen.ARZ import ARZ

        energy = 1 * units.EeV
        theta = 57 * units.deg # Viewing angle w.r.t. shower axis
        N = 2048 # Number of samples 
        dt = 0.2 * units.ns # Time binning
        shower_type = 'HAD' # 'HAD' or 'EM'
        R = 1 * units.km # Distance to shower vertex
        n_index = 1.78 # refractive index
        model = 'ARZ2019'
        same_shower = True

        cARZ = ARZ.ARZ()
        trace = cARZ.get_time_trace(shower_energy, theta, N, dt, shower_type, n_index, R, shift_for_xmax=False)

In this case, the trace object is a 2D array where the first dimension controls the electric field coordinates and the second gives the time dependence. The electric field is given in a spherical frame (:math:`E_r`, :math:`E_{\theta}`, :math:`E_{\phi}`), where the theta unit vector is perpendicular to the line of sight between the observer and *the shower maximum* and lies on the plane defined by the shower axis and the observer. The radial unit vector lies on the line of sight between observer and shower maximum. The phi unit vector is the cross product of the radial vector times the theta vector. Important: each time the function ``get_time_trace`` is called, a new, different shower is taken from the shower library. If we want to use the same shower, the keyword argument ``same_shower = True`` can be used.

Brief explanation of some of the most obscure parameters of the function:

    .. code-block:: Python

        """
        shift_for_xmax: bool (default False)
            if True the observer position is placed relative to the position of the shower 
            maximum, if False it is placed with respect to (0,0,0) which is the start of 
            the charge-excess profile
        """

If ``shift_for_xmax`` is ``True``, the distance (:math:`R`) and angle (:math:`\theta`) fed to the function are referred to the shower maximum instead of the shower vertex. This makes sense, as it is the shower maximum the part that also emits the most electric field, and therefore the field scales with the distance to the shower maximum and the Cherenkov angle should be measured with respect to the shower maximum as well. We only recommend it to be set to False when it is absolutely necessary that the vertex becomes the reference point.

    .. code-block:: Python

        """
        same_shower: bool (default False)
            if False, for each request a new random shower realization is chosen.
            if True, the shower from the last request of the same shower type is used. 
            This is needed to get the Askaryan signal for both ray tracing solutions from 
            the same shower.
        iN: int or None (default None)
            specify shower number
        output_mode: string
            * 'trace' (default): return only the electric field trace
            * 'Xmax': return trace and position of xmax in units of length
            * 'full' return trace, depth and charge_excess profile
        """

The user can choose to have a tuple with the trace and the distance between the vertex and shower maximum if ``output_mode`` is ``'Xmax'``. If it is ``'full'``, the function returns a tuple with three elements: the trace, the shower profile depths in units of mass per area, and shower charge excess in number of excess negative particles. Keep in mind that in NuRadioReco, the energy unit is the electronvolt and the kilogram is defined as J s\ :sup:`2` m\ :sup:`-2`, which makes densities hard to read with default units. We
recommend to always divide all variables by the units the user wants to display them on, and even more for density units.

Validity of the parameterisations and the ARZ model
------------------------------------------------------
The ARZ model is valid as long as the minimum distance between shower and observer is much larger than the minimum observation wavelength of interest. When this happens we say that the observer is in the far field (in an electromagnetic sense) (`see paper <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.87.023003>`__):

    1. :math:`kR >> 1`,

with k the wavenumber and R the distance between shower and observer. The Fresnel condition has also to be fulfilled, which can be expressed as:

    2. :math:`k L^2R \sin^2(\theta) << 1`,

with k the wavenumber, L the length of the shower, R the distance between shower and observer, and :math:`\theta` the viewing angle with respect to the shower axis. In practice, these two conditions apply to almost any neutrino shower in our simulations.

The integrals in the ARZ model present serious instabilities when the observer is near the axis or near 90 degrees (perpendicular to the shower axis). Along these directions, coherence is almost non-existent, and therefore the electric field is much lower than near the Cherenkov angle. However, these numerical instabilities can create artificial peaks and trigger our detector. **ARZ should not be used with viewing angles lower than 30 or greater than 80**. To that effect, we can limit the ``delta_C_cut`` in the configuration file so that we don't consider events 25 degrees away from the Cherenkov angle.

For the parameterisations to be valid, besides Eqs. 1) and 2), we have one extra condition. The shower has to be far away from the observer, such as all the different parts of the shower are seen by the observer with the same viewing angle. In other words, the shower has to be approximated as a point-like region. Some people call this also the far-field approximation, to compare with the one defined above. So it is good to always ask if they mean far away with respect to the wavelength or far away with respect to the shower size.

The parameterisations and the ARZ model, just like the ZHS Monte Carlo, have been created for a homogeneous medium. However, for experiments like ARIANNA or RNO, the ice layer cannot be considered homogeneous. In the atmosphere, even for relatively inclined showers, the electromagnetic waves are not bent because of the refraction and keeping track of the different speed of light at each height is enough. However, in ice, we also need to calculate how much the rays bend near the surface (100 m for Greenland and 200 m for South Pole). That is why the signal generation module has to be combined with a ray tracing module, such as the one in SignalProp in NuRadioMC. However, knowing how they should be combined and justifying it is not easy. In NuRadioMC, we assume that we can calculate the electric field in a homogeneous medium given by the refractive index in the vicinity of the shower. The ray tracing module is called to know the distance travelled by the refracted ray and how much it bends. This information is used to get a corrected distance and a corrected viewing angle that are then fed to the signal generation module to calculate the electric field. The simulation module makes sure that the geometry and the rotations are correct. But we must keep three things in mind:

* We are using the index in the vicinity of the shower while codes like ZHAireS or CoREAS use the average index along the ray path. In the atmosphere, this path is a straight line, but the different speeds change the propagation times and therefore also change the coherence pattern. This, in turn, changes the Cherenkov cone, which is really important. 
  It can be argued that we should use the **average** refractive index **along the ray path**. This is not so relevant if the antenna is located 100 m deep in ice, and the ray comes from below and on a direct path, but it can change the emission a lot for shallow channels and reflected trajectories. To give an idea of how much this could affect, keep in mind that the index of deep ice is 1.78 and the Cherenkov angle is 55.8 degrees. If the average index along the path is 1.70 instead, the Cherenkov angle changes to 54 degrees. These 3 degrees of difference can hinder our reconstruction accuracy. For shallow channels or reflected trajectories, if the average index along the path is 1.5, the Cherenkov angle would be 48.2 degrees.    
* There is a focusing correction implemented in the simulation module. In a non-homogeneous optical medium, the rays from a source with slightly different trajectories can converge at the same point, which causes regions with more concentration of electric field. Next to the shadow zone, where the rays cannot reach the observer, there is a region called the caustic, where the concentration is the largest.
* The combination of ray tracing (bending plus speed changes) and focusing is thought to be a good approximation to the actual electric field. However, we have seen that it is not entirely clear what index to use. It would be nice to be able to settle this debate with equations, but the solutions for Maxwell's equations in a non-homogeneous medium like ours have not been calculated yet. To prove that is harder than it looks, let us take one of Maxwell's equations in a medium:

:math:`\nabla\cdot\boldsymbol{D} = \rho_f`,

with :math:`\boldsymbol{D}` the electric displacement and :math:`\rho_f` the free density charge. Let us express this equation in terms of the electric field :math:`\boldsymbol{E}` and then in terms of the potentials.

:math:`\nabla\cdot\boldsymbol{D} = \rho_f = \nabla\cdot(\epsilon\boldsymbol{E}) = \epsilon\nabla\cdot\boldsymbol{E} + 
\boldsymbol{E}\nabla\epsilon`,

where we have used the definition of electric displacement with a position-dependent permittivity, :math:`\epsilon = \epsilon(\boldsymbol{x})`. Now, in terms of potentials:

:math:`\rho_f = \epsilon(-\nabla^2\Phi - \partial_t\nabla\cdot\boldsymbol{A}) +
(-\nabla\Phi - \partial_t\boldsymbol{A}) \cdot\nabla\epsilon`,

where we have only used :math:`\boldsymbol{E} = - \nabla\Phi - \partial_t\boldsymbol{A}`. Let us use the Lorenz gauge condition:

:math:`\nabla\cdot\boldsymbol{A} = -\mu\epsilon\partial_t\Phi`,

which leads us to:

:math:`\rho_f = -\epsilon(\nabla^2\Phi - \partial_t^2\Phi) + (-\nabla\Phi -\partial_t\boldsymbol{A}) \cdot\nabla\epsilon = -\epsilon(\nabla^2\Phi - \partial_t^2\Phi) + \boldsymbol{E}\cdot\nabla\epsilon`.

This reduces to the standard wave equation if we neglect the term :math:`\boldsymbol{E}\cdot\nabla\epsilon`. This is the approximation that is usually made (implicitly) in papers like `this one <https://dx.doi.org/10.1103/PhysRevLett.123.091102>`__. While it's a good starting point, we cannot claim to have a complete solution when neglecting this term. The gradient of the permittivity doesn't seem negligible in shallow ice, and also this term is coupled to the electric field, which means it is larger for those regions where the electric field is larger. Another problem is that when neglecting this term, refraction disappears and we have to put it back ad hoc, as we usually do. But it is in principle not clear what we have lost along the way by removing the permittivity gradient and putting it back with ray tracing, although it does seem physically sound as a first-order approximation. The best way to settle this debate is with a proper finite differences in time domain (FDTD) method.

Timing
------
Knowing the way timing works in NuRadioMC is crucial for reconstruction applications. As of now, the calculation of times is an interplay between the event generator, the signal generator, and the signal propagator modules.

    1. The event generator creates a vertex time, assuming that the first neutrino interaction happens at ``t = 0``, and the time for subsequent interactions are given by the time of flight.
    2. The signal generator creates a trace. For every model, the **middle of the trace** corresponds to the time when the **signal from the vertex arrives** at the observer. If the parameterisations are used, the timing will not be accurate to nanosecond or subnanosecond level, since these models do not contain phase information. **Parameterisations must NOT be used when accurate timing is needed.**
    3. The signal propagation module computes the time it takes for the wave to get from vertex position to observer.

The simulation module takes the vertex interaction time (point 1), adds the propagation time (point 3) and substracts half of the trace window to obtain the observer time for the first point of the trace. This time is then used as the channel trace starting time.

In short, the times obtained for each channel trace are calculated assuming that the first interaction happens at ``t = 0``, and the time array obtained with the method ``channel.get_times()`` are consistent with this description.

Using the same shower. Random seed
----------------------------------
Two of the most relevant models have randomness added to simulate shower to shower fluctuations: the Alvarez2009 and the ARZ2020 (or 2019) model.

The randomness in the Alvarez2009 model comes from a crude simulation of the LPM effect for electromagnetic showers. The constant k:sub:`L` is obtained from a log-gaussian distribution to simulate the stretching of the shower due to LPM. This stretching leaves the EM shower with the same smooth profile, only extended along a larger length, unlike a true LPM shower profile which should be bumpy because of the stochastic interactions. Nevertheless, it is a decent first approximation for illustration purposes.

The ARZ models give different results each time they are called because the shower profile taken from the shower library changes. This mimics shower-to-shower fluctuations in a rigorous way.

The problem arises when we want to use the same shower to calculate the field for different positions, like different channels. All we have to do is call any of the ``get_time_trace`` methods, whether in the Askaryan module, the paremeterisations module or the ARZ module, with the argument:

    .. code-block:: Python

        same_shower = True

If we want to draw a different shower, it suffices to call the ``get_time_trace`` once again with ``same_shower`` set to ``False`` and then set it again to ``True``.

If the user wants to use a specific shower to reproduce some previous results or to make them reproducible in the future,
the numpy random seed can be fixed right before calling the  ``get_time_trace`` method.

    .. code-block:: Python

        import numpy as np
        my_seed = 42
        np.random.seed(my_seed)

FFT normalisation
--------------------
FFT normalisation is a confusing subject to most people that that have worked with FFT algorithms. For the sake of speed, FFT algorithms work without any notion of dimensions or intervals between the points of an array, which combined with many other things, allows for a much faster transform, hence the first F for fast. The issue is, when using an FFT to transform from time to frequency, for instance, the FFT does not tell us the dimensions of the resulting transform or the frequencies. That is left for the user, so the fast part seems to be when calculating the transform and not when one has to interpret it, where it is really easy to commit a mistake and waste more time, effectively turning our FFTs into SFTs (Slow Fourier Transforms) or JWFTs (Just Wrong Fourier Transforms). The problem is only aggravated when transforming back, as some implementations present an extra N factor, the number of samples.

With the FFT implementation in numpy, it is enough to multiply the result by the sampling rate to get the Fourier transform in the trace dimensions per frequency unit. It uses a standard normalisation with factor 1. In our case, we are using ``numpy.rfft``, the real transform, instead of the general complex ``numpy.fft``. This implementation ignores the negative frequencies, so if we want to get the whole energy of a signal integrating on the positive frequencies only, we have to multiply our transform by a factor of square root of 2.

We have written a convenient FFT wrapper in NuRadioReco.utilities.fft. This wrapper has two functions, ``time2freq``, to transform from time to frequency, and ``freq2time``, for the inverse operation. The factors of square root of 2 and the dimensions of the function in frequency space are automatically taken into account. These functions need as arguments both the trace or spectrum, and the sampling rate. ``time2freq`` returns the spectrum in the input trace units per GHz. So, if a voltage in volts is passed as input, ``time2freq`` will return the spectrum in V/GHz.

    .. code-block:: Python

        from NuRadioReco.utilities import fft
        from NuRadioReco.utilities import units
        from NuRadioMC.SignalGen.askaryan import get_time_trace
        import numpy.fft

        sampling_time = 0.1 * units.ns
        # we define an electric field trace with any arguments and model we fancy
        trace = get_time_trace(..., sampling_time, ...)
        sampling_rate = 1/sampling_time
        spectrum = fft.time2freq(trace, sampling_rate)
        # The resulting spectrum has dimensions of V/m/GHz and we can get the whole energy
        # by integrating positive frequencies only.

        # The frequencies can be obtained with numpy.fft.rfftfreq:
        n_samples = len(trace)
        frequencies = numpy.fft.rfftfreq(n_samples, sampling_time)
        # Now we could perform operations in frequency domain

        # Let us transform back to time
        trace_back_in_time = fft.freq2time(spectrum, sampling_rate)
        # trace and trace_back_in_time are equal if no operation has been performed on spectrum