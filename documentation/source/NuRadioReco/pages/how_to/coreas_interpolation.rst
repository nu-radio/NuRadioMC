Using the cosmic-ray pulse interpolator in NRR
==============================================

The cosmic ray pulse interpolator was developed by Arthur Corstanje
and is described in `this paper <https://doi.org/10.1088/1748-0221/18/09/P09005>`__.
This algorithm was implemented in a separate repository under the NuRadio family,
and is available at `this link <https://github.com/nu-radio/cr-pulse-interpolator>`__ .
Since NRR version 3.1.0, this module is available in the NRR framework through an
interface that makes it easy to combine NRR Event objects with the interpolator.

This module is called `coreasInterpolator` and can be found in the
`NuRadioReco.modules.io.coreas.coreasInterpolator` module. It is capable of both signal
(i.e. 3D electric field traces) as well as fluence (i.e. a single value) interpolation.

Initialising the interpolator
-----------------------------

In order to perform the interpolation, you will need a so-called starshape simulation.
This is an HDF5 file of a CoREAS simulation where the antennas have been placed
along 8 arms, at fixed radial distances from the shower core.

.. note::
    While we took some care to ensure that the core of the simulation is taken
    into account everywhere, the module has never been tested with simulations
    that have a core not at (0, 0). If for some very obscure reason you need to
    use a simulation with a core not at (0, 0), please take extra care and check
    the interpolation results before using them.

The first step is to read in this simulation using the :func:`read_CORSIKA7() <NuRadioReco.modules.io.coreas.coreas.read_CORSIKA7>` function
from the `NuRadioReco.modules.io.coreas.coreas` module. This returns an Event object
containing the electric field traces and all the necessary information about the
simulation.

The next step is to create the interpolator object. The initialisation function
takes in the Event we created in the previous step. After the interpolator has been
initialised, we can check that it has been set up correctly by for example looking
at the `zenith` and `azimuth` properties.

After that, we initialize either the signal or fluence interpolation. This is done
by calling the :func:`initialize_efield_interpolator() <NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator.initialize_efield_interpolator>`
or :func:`initialize_fluence_interpolator() <NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator.initialize_fluence_interpolator>`.
respectively. Both functions take in arguments that can be used to configure the
interpolation. For example, to get more output from the interpolator, you can set
the `verbose` argument to `True`. To know what arguments are available, you can
check the initialisation functions in the `cr-pulse-interpolator <https://github.com/nu-radio/cr-pulse-interpolator>`__ repository.

The complete procedure looks like this:

  .. code-block:: python

    from NuRadioReco.modules.io.coreas.coreas import read_CORSIKA7
    from NuRadioReco.modules.io.coreas.coreasInterpolator import coreasInterpolator

    # read in the starshape simulation
    event = read_CORSIKA7('path/to/simulation.hdf5')

    # create the interpolator object
    interpolator = coreasInterpolator(event)

    # initialize the signal interpolation
    interpolator.initialize_efield_interpolator(30 * units.MHz, 80 * units.MHz)
    # OR interpolator.initialize_fluence_interpolator()

When initialising a signal interpolator, we need to specify the frequency range
over which we want to interpolate. This means that the interpolated signals that
we retrieve afterwards will be filtered to this frequency range.

Interpolating signals
---------------------

Once the interpolator has been initialised, we can obtain the interpolated signals at
any location. This is done by calling the :func:`get_interp_efield_value() <NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator.get_interp_efield_value>`
method, if we are
working with a signal interpolator, or the
:func:`get_interp_fluence_value() <NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator.get_interp_fluence_value>`
method in case of fluence interpolation.
Both methods take in the position on the ground where you want
the interpolated value. If you want to shift the core of the simulation around, you should
simply subtract the new core position from the antenna location.

The :func:`get_interp_efield_value() <NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator.get_interp_efield_value>`
method returns the electric field in the same coordinate
system as the input electric fields. If you wish to have them in a different reference
system, you can use the `cs_transform` argument.

.. note::
    While the interpolator can be configured to also extrapolate outside of the
    input region of the starshape, this is not recommended. Therefore, by default
    the interpolator will return zero signals when the requested location is outside
    of the starshape and some constant trace when the requested location is inside
    the original starshape.

What happens when you initialise a signal interpolator?
-------------------------------------------------------

In the paper that introduced the Fourier interpolation method, the authors noted some
important limitations.

First, when the geomagnetic angle (i.e. the angle between the magnetic field vector and
the shower axis) is smaller than 15 degrees, the interpolation is not reliable. Therefore,
in our implementation the interpolator switches to a different method in this case. Namely,
it will simply return the electric field of the nearest antenna location. However, the arrival
time of the signals are still interpolated, so the returned signal start time should still
reflect the geometry. This happens automatically and is logged. You can also verify if the
interpolator is using nearest neighbour interpolation by checking if
`interpolator.efield_interpolator_initialized` is ``False``.

Secondly, it is important to ensure that the electric field polarisations are **not** aligned
with the :math:`\vec{v} \times \vec{B}` and :math:`\vec{v} \times \vec{v} \times \vec{B}` vectors.
Otherwise, when going around the circle, one of components will have a near-zero amplitude at some
point. At this point calculating the phases becomes impossible, which leads to difficulties in
the interpolation. A first mitigation for this, is to provide the electric fields in a on-sky
coordinate system. This is done by default when reading in a simulation using the `read_CORSIKA7()`
function. Next, when initialising the interpolator, a check if performed automatically to see
if the shower is coming from close to zenith or along the north-south axis. If so, the electric
field polarisations are rotated by 45 degrees. When requesting an electric field using the
:func:`get_interp_efield_value() <NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator.get_interp_efield_value>`
method, this rotation is automatically undone. So the returned electric
field is in the same coordinate system as the input electric fields.

Some tips to work with fluence interpolation
--------------------------------------------

In order to interpolate the fluence, it needs to be stored in some electric fields parameter
( :ref:`label_parameter_storage` ). To easily set this for all electric fields used by the
interpolator, you can use its
:func:`set_fluence_of_efields <NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator.set_fluence_of_efields>`
method. You can pass in any function
that you want that takes it the 3-dimensional electric field traces and return the value that
needs to be stored in the parameter.

To quickly check whether the values make sense, you can use the
:func:`plot_fluence_footprint <NuRadioReco.modules.io.coreas.coreasInterpolator.coreasInterpolator.plot_fluence_footprint>` method
to see how the footprint looks like.
