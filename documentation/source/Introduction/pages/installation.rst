Installing NuRadioMC / NuRadioReco
==================================

Requirements
------------
In order to use ``NuRadioMC`` / ``NuRadioReco``, please ensure you are using a version of Python ``>=3.7``, and a UNIX operating system (linux or MacOS).
If you are using Windows, consider installing the `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install>`_.

.. Note::

  **We highly recommend installing NuRadioMC inside a** `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_.
  You can either use ``python3 -m venv name_of_venv``
  or use a virtual environment manager like `conda <https://anaconda.org/anaconda/python>`_.

Installation using ``pip``
--------------------------
``NuRadioReco`` is a subpackage of ``NuRadioMC``, so both are installed at once using ``pip``:

.. code-block:: Bash

  pip install NuRadioMC

NuRadioMC/NuRadioReco will then be available from Python using ``import NuRadioMC`` and ``import NuRadioReco``, respectively.
This installs all core dependencies. Some features require additional packages, which are grouped into
:ref:`options <Introduction/pages/installation:Optional Dependencies>` and can be installed by appending ``[option]``,
i.e. ``pip install NuRadioMC[option]``. One can also use ``[all]`` to install all (non-development) dependencies.

.. Important::

  Some optional dependencies cannot be installed using pip and
  :ref:`have to be installed manually <Introduction/pages/installation:Not pip-installable packages>`.

.. Note:: This is the release version of NuRadioMC. If you want the latest (development) version, use

  .. code-block::

    pip install git+https://github.com/nu-radio/NuRadioMC.git

  instead, or install it manually (see below).

Development version
-------------------
If you want the most recent, in-development version of ``NuRadioMC``, or intend to :doc:`contribute to its development </Introduction/pages/contributing>`,
you can get it via `the NuRadioMC github <https://github.com/nu-radio/NuRadioMC.git>`__:

.. code-block:: Bash

  git clone https://github.com/nu-radio/NuRadioMC.git

If you don't already have it installed, you should `install Git <https://git-scm.com/>`_.

To install NuRadioMC and its dependencies, use the `pip editable install <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`__.
Navigate to the ``NuRadioMC`` folder and run:

.. code-block:: bash

  cd NuRadioMC/
  pip install -e .[dev]
  pre-commit install

(note the ``-e`` flag!). This will install the core dependencies, as well as the optional ``dev`` dependencies (use ``[dev,all]`` instead to also install all optional dependencies),
and tell ``python`` to look for ``NuRadioMC`` and ``NuRadioReco`` in this folder, so that you can edit and contribute to the codebase while using it.
The last line, ``pre-commit install`` installs a git hook using `pre-commit <https://pre-commit.com>`__. **This is highly recommended for developers as it
helps to keep the repository clean from accidentally added large files**. More details are given :ref:`here <Introduction/pages/contributing:Installing NuRadioMC for developers>`.

.. Note::

  Users of ``zsh`` (the default shell on MacOS) have to wrap the argument in quotes, i.e. ``pip install -e '.[dev]'``,
  as ``zsh`` otherwise interprets the square brackets itself.

Optional Dependencies
---------------------

The core installation of NuRadioMC deliberately stays lightweight: several features rely on additional packages
that are not installed by default. These are grouped into *options* (also called "extras"), which are appended in
square brackets when installing NuRadioMC, e.g.

.. code-block:: bash

  pip install NuRadioMC[proposal]                 # a single option
  pip install NuRadioMC[proposal,galacticnoise]   # several options at once
  pip install -e .[dev,proposal]                  # the same for a development install

The available options are:

.. list-table::
  :header-rows: 1
  :widths: 20 30 50

  * - Option
    - Installs
    - Needed for
  * - ``[proposal]``
    - ``proposal``
    - Propagation of secondary leptons, :mod:`NuRadioMC.EvtGen.NuRadioProposal`
  * - ``[galacticnoise]``
    - ``pygdsm``, ``pylfmap``, ``healpy``
    - Adding galactic noise, :mod:`NuRadioReco.modules.channelGalacticNoiseAdder`
  * - ``[muon-flux]``
    - ``MCEq``, ``crflux``
    - Atmospheric muon flux calculations, :mod:`NuRadioMC.utilities.muon_flux`
  * - ``[cr_interpolator]``
    - ``cr-pulse-interpolator``
    - Interpolation of CoREAS star-shape simulations, :mod:`NuRadioReco.modules.io.coreas.coreasInterpolator`
  * - ``[minimizers]``
    - ``iminuit``, ``scikit-optimize``, ``noisyopt``
    - Additional minimizers in :mod:`NuRadioReco.utilities.minimization`
  * - ``[dev]``
    - ``pre-commit``, ``Sphinx``, ``sphinx-rtd-theme``, ``numpydoc``
    - Contributing to NuRadioMC and building the documentation locally
  * - ``[all]``
    - all of the above except ``[dev]``
    - Convenience option to get all user-facing features

Note that option names are normalised by ``pip``, so ``[cr_interpolator]`` and ``[cr-interpolator]`` are equivalent.
To get everything, including the development dependencies, use ``pip install NuRadioMC[all,dev]``.
A few features additionally depend on packages that are not part of any option: they either
:ref:`cannot be installed via pip <Introduction/pages/installation:Not pip-installable packages>` or are only needed
for :ref:`specific detectors or data formats <Introduction/pages/installation:Other optional packages>`.

- ``[proposal]``

  `PROPOSAL <https://github.com/tudo-astroparticlephysics/PROPOSAL>`__ is a lepton propagation code. It is needed to use the
  :mod:`NuRadioMC.EvtGen.NuRadioProposal` module, which simulates the secondary interactions of muons and taus, i.e. the
  showers that these leptons induce along their path through the ice. Without it, NuRadioMC only simulates the shower
  produced at the neutrino interaction vertex itself.

  .. code-block:: bash

    pip install proposal==7.6.2

  Note that the pip installation for this version of proposal may not work on all systems, in particular:

  - conda cannot be used on all systems (eg. on Mac), in that case use a python venv, see details `here <https://github.com/tudo-astroparticlephysics/PROPOSAL/issues/209>`__

  - if the linux kernel is too old (eg. on some computing clusters), refer to `this step-by-step guide <https://github.com/tudo-astroparticlephysics/PROPOSAL/wiki/Installing-PROPOSAL-on-a-Linux-kernel---4.11>`_

- ``[galacticnoise]``

  The :mod:`channelGalacticNoiseAdder <NuRadioReco.modules.channelGalacticNoiseAdder>` and
  :mod:`efieldGalacticNoiseAdder <NuRadioReco.modules.efieldGalacticNoiseAdder>` modules add the diffuse emission of
  the galaxy to simulated traces. This is the dominant noise source for detectors at frequencies below a few hundred
  MHz, e.g. for air-shower detection. The sky models are provided by `PyGDSM <https://github.com/telegraphic/pygdsm>`_
  (GSM, GSM2016, LFSM, ...); ``pylfmap`` provides the additional LFmap model used by LOFAR for calibration purposes,
  and ``healpy`` is used to handle the sky maps.

  .. code-block:: Bash

    pip install pygdsm pylfmap healpy

- ``[muon-flux]``

  ``MCEq`` solves the cascade equations for the atmosphere and ``crflux`` provides parametrisations of the
  cosmic-ray flux. Together they are used by :mod:`NuRadioMC.utilities.muon_flux` to calculate the flux of
  atmospheric muons at the surface, which is the main background for in-ice radio detectors at PeV energies.

  .. code-block:: bash

    pip install MCEq crflux

- ``[cr_interpolator]``

  Installs the cosmic-ray pulse interpolator from https://github.com/nu-radio/cr-pulse-interpolator. CoREAS air-shower
  simulations are usually produced on a star-shaped pattern of observer positions. The
  :mod:`coreasInterpolator <NuRadioReco.modules.io.coreas.coreasInterpolator>` uses this package to interpolate the
  simulated pulses to arbitrary positions in between, so that a single CoREAS simulation can be reused for many
  detector positions or shower cores.

  .. code-block:: bash

    pip install cr-pulse-interpolator

- ``[minimizers]``

  The :mod:`Minimizer <NuRadioReco.utilities.minimization>` class provides a common interface to different
  minimization algorithms used in reconstruction. Beyond the ``scipy`` minimizers (which are always available), it can
  use `iminuit <https://scikit-hep.org/iminuit/>`__ (MIGRAD/MINOS, including proper uncertainty estimation),
  `scikit-optimize <https://scikit-optimize.github.io/>`__ (Bayesian/global optimization) and
  `noisyopt <https://github.com/andim/noisyopt>`__ (minimization of noisy objective functions).

  .. code-block:: bash

    pip install iminuit scikit-optimize noisyopt

- ``[dev]``

  For developers, we use `pre-commit <https://pre-commit.com>`__ to prevent the accidental addition of large files that would clutter the repository, as well as run some simple
  code formatting checks (see :ref:`here <Introduction/pages/contributing:Installing NuRadioMC for developers>` for more details):

  .. code-block:: bash

    pip install pre-commit

  The documentation is created using `Sphinx <https://www.sphinx-doc.org>`_. We use the ``readthedocs`` theme, and the ``numpydoc`` format is used in our docstrings.
  These dependencies are needed only if you want to generate the documentation locally - the `online documentation <https://nu-radio.github.io/NuRadioMC/main.html>`_ is generated by a Github action automatically.
  Note that we use the `sphinx autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc>`_
  feature, which tries to import all modules it documents. So if you are missing some optional dependencies, it will not generate correct documentation for all the code.

  .. code-block:: Bash

    pip install sphinx sphinx_rtd_theme numpydoc

Not pip-installable packages
----------------------------

- To speed up the :mod:`analytic ray tracing module <NuRadioMC.SignalProp.analyticraytracing>`, `GSL <https://www.gnu.org/software/gsl/>`_ needs
  to be installed, and ``$GSL_DIR`` should point at the correct installation folder. On Linux, GSL can be installed using

  .. code-block:: bash

    sudo apt-get install libgsl-dev

  (On MacOS, use ``brew install gsl`` instead - you may have to install `homebrew <https://brew.sh/>`_ first).
  With GSL installed, compile the CPP ray tracer by navigating to ``NuRadioMC/NuRadioMC/SignalProp``
  and running the included ``install.sh`` script.
- To use the :mod:`RadioPropa numerical ray tracing <NuRadioMC.SignalProp.radioproparaytracing>` module, ``radiopropa`` needs to be installed.
  The radiopropa github, with installation instructions, can be found `here <https://github.com/nu-radio/RadioPropa>`__.
- To read ARIANNA files, `Snowshovel <https://arianna.ps.uci.edu/mediawiki/index.php/Local_DAQ_Instructions>`_ needs to be installed.
- To read ARA files, `ARA ROOT <http://www.hep.ucl.ac.uk/uhen/ara/araroot/branches/3.13/index.shtml>`_ needs to be installed.

Other optional packages
-----------------------

These packages can be installed with pip, but are not part of any option because they are only needed for specific
detectors or data formats.

- `mattak <https://github.com/RNO-G/mattak>`__ is required to open RNO-G root files:

  .. code-block:: bash

    pip install git+https://github.com/RNO-G/mattak

  Optionally, to filter RNO-G data (during read in) the `RNO-G run table database <https://github.com/RNO-G/rnog-runtable>`__
  can be used. Note that this requires membership of the RNO-G Github organisation (not public):

  .. code-block:: bash

    pip install git+ssh://git@github.com/RNO-G/rnog-runtable.git

- To use a detector description stored in an SQL database (:mod:`NuRadioReco.detector.detector_sql`), install
  `MySQL <https://www.mysql.com/>`_ and mysql-connector-python:

  .. code-block:: Bash

    pip install mysql-connector-python
