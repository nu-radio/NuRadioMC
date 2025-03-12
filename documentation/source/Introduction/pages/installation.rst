Installing NuRadioMC / NuRadioReco
==================================

Requirements
------------
In order to use ``NuRadioMC`` / ``NuRadioReco``, please ensure you are using a version of Python ``>=3.7``, and a UNIX operating system (linux or MacOS).
If you are using Windows, consider installing the `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install>`_.

Installation using ``pip``
--------------------------
As of version ``2.0.0``, ``NuRadioReco`` is now a subpackage of ``NuRadioMC``. Both ``NuRadioMC`` and ``NuRadioReco`` can therefore be installed
using ``pip``:

    .. code-block:: Bash

      pip install NuRadioMC

NuRadioMC/NuRadioReco will then be available from Python using ``import NuRadioMC`` and ``import NuRadioReco``, respectively.
The pip installation will install all core dependencies. Some :ref:`optional dependencies <Introduction/pages/installation:Optional Dependencies>`
can be installed by appending ``[option]``, i.e. ``pip install NuRadioMC[option]``. One can also use ``[all]`` to install all (non-development) dependencies.

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

.. Note::

  **We highly recommend installing NuRadioMC inside a** `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_.
  You can either use ``python3 -m venv name_of_venv``
  or use a virtual environment manager like `conda <https://anaconda.org/anaconda/python>`_.

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

Manual installation
-------------------

Pip-installable dependencies
____________________________

To install all (optional and non-optional) dependencies available in pip at once, use the command

.. code-block:: bash

  pip install numpy scipy matplotlib astropy tinydb tinydb-serialization aenum h5py mysql-connector-python pymongo dash plotly toml peakutils future radiotools filelock mattak git+https://github.com/telegraphic/pygdsm pylfmap MCEq crflux git+https://github.com/nu-radio/cr-pulse-interpolator

Only for developers, we also recommend

.. code-block:: bash

  pip install Sphinx sphinx-rtd-theme numpydoc pre-commit

which are used to locally compile the documentation, and to perform :ref:`some safety checks <Introduction/pages/contributing:Installing NuRadioMC for developers>` when contributing new code.

Note that some optional dependencies are not pip-installable and need to be 
:ref:`installed manually <Introduction/pages/installation:Not pip-installable packages>`

Core Dependencies
^^^^^^^^^^^^^^^^^
- toml:
  
  .. code-block:: bash

    pip install toml
    
- radiotools:

  .. code-block:: bash

    pip install radiotools

- numpy:

  .. code-block:: Bash

    pip install numpy

- scipy:

  .. code-block:: Bash

    pip install scipy

- matplotlib:

  .. code-block:: Bash

    pip install matplotlib

- astropy:

  .. code-block:: Bash

    pip install astropy

- tinydb:
  tinydb version 4.1.1 or newer is required.

  .. code-block:: Bash

    pip install tinydb tinydb-serialization

- Advanced enum:

  .. code-block:: Bash

    pip install aenum

- h5py to open HDF5 files:

  .. code-block:: Bash

    pip install h5py

- filelock:

  .. code-block:: Bash

    pip install filelock

- For `MongoDB <https://www.mongodb.com>`_ databases install:

  .. code-block:: Bash

    pip install pymongo

- To use the :ref:`Event Display <NuRadioReco/pages/event_display:Event Display>` you need plotly and dash:

  .. code-block:: Bash

    pip install dash
    pip install plotly

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

These packages are recommended to be able to use all of NuRadioMC/NuRadioReco's features.
They can be installed by including adding ``[option]`` when installing NuRadioMC. Alternatively,
use ``pip install nuradiomc[all]`` to install all optional dependencies (or ``[all,dev]`` to also install development dependencies).

- ``[RNO-G]``

  `mattak <https://github.com/RNO-G/mattak>`__ is required to open RNO-G root files:

  .. code-block:: bash

    pip install mattak

- ``[rno-g-extras]``

  Optionally, to filter RNO-G data (during read in) the `RNO-G run table database <https://github.com/RNO-G/rnog-runtable>`__
  can be used. Note that this requires membership of the RNO-G Github organisation (not public):

  .. code-block:: bash

    pip install git+ssh://git@github.com/RNO-G/rnog-runtable.git

- ``[proposal]``

  ``proposal`` is needed to use :mod:`NuRadioMC.EvtGen.NuRadioProposal` module (simulating secondary particles):

  .. code-block:: bash

    pip install proposal==7.6.2

  Note that the pip installation for this version of proposal may not work on all systems, in particular:

  - conda cannot be used on all systems (eg. on Mac), in that case use a python venv, see details `here <https://github.com/tudo-astroparticlephysics/PROPOSAL/issues/209>`__

  - if the linux kernel is too old (eg. on some computing clusters), refer to `this step-by-step guide <https://github.com/tudo-astroparticlephysics/PROPOSAL/wiki/Installing-PROPOSAL-on-a-Linux-kernel---4.11>`_
  
- ``[galacticnoise]``

  To use the channelGalacticNoiseAdder, you need the `PyGDSM <https://github.com/telegraphic/pygdsm>`_ package.
  Some additional galactic noise models used by LOFAR for calibration purposes are provided by ``pylfmap``.

  .. code-block:: Bash

    pip install git+https://github.com/telegraphic/pygdsm pylfmap

- ``[muon-flux]``

  Needed for some muon flux calculations

  .. code-block:: bash

    pip install MCEq crflux

- ``[cr-interpolator]``

  Installs the cosmic-ray interpolator from https://github.com/nu-radio/cr-pulse-interpolator,
  used to interpolate cosmic-ray air-shower emission from CoREAS star-shaped patterns:

  .. code-block:: bash

    pip install git+https://github.com/nu-radio/cr-pulse-interpolator

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

- Some debug plots need peakutils:

  .. code-block:: Bash

    pip install peakutils

- For SQL databases install `MySQL <https://www.mysql.com/>`_ and mysql-connector-python:

  .. code-block:: Bash

    pip install mysql-connector-python


Not pip-installable packages
____________________________

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
