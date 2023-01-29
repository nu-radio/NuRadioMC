Installing NuRadioMC / NuRadioReco
==================================

Requirements
------------
In order to use ``NuRadioMC`` / ``NuRadioReco``, please ensure you are using a version of Python ``>=3.6``, and a UNIX operating system (linux or MacOS).
If you are using Windows, consider installing the `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install>`_.

Installation using Pip
-------------------------
As of version ``2.0.0``, ``NuRadioReco`` is now a subpackage of ``NuRadioMC``. Both ``NuRadioMC`` and ``NuRadioReco`` can therefore be installed
using ``pip``:

    .. code-block:: Bash

      pip install NuRadioMC

NuRadioMC/NuRadioReco will then be available from Python using ``import NuRadioMC`` and ``import NuRadioReco``, respectively.
The pip installation will also install all core dependencies. 

.. Important::

  Some optional dependencies cannot be installed using pip and 
  :ref:`have to be installed manually <Introduction/pages/installation:Not pip-installable packages>`.

.. Note:: If you want the current version or you want to contribute to NuRadioReco, you need to install it manually.

Development version
---------------------------
The most recent version of ``NuRadioMC`` is available on `github <github.com>`_. 
It can be downloaded manually from the `repository website <https://github.com/nu-radio/NuRadioMC.git>`_,
or cloned using ``git``

.. code-block:: Bash

  git clone https://github.com/nu-radio/NuRadioMC.git

If you don't already have it installed, you should `install Git <https://git-scm.com/>`_. 
To set up NuRadioMC and install the required dependencies, navigate to the ``NuRadioMC`` folder and run the python installer provided:

.. code-block:: bash

  cd NuRadioMC/
  python3 install_dev.py

This will launch an interactive installer that allows you to install all core and some optional dependencies, as well as setting up some git settings
in case you want to contribute to NuRadioMC. **We highly recommend installing NuRadioMC inside a** `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_. 
You can either use ``python3 -m venv name_of_venv``
or use a virtual environment manager like `conda <https://anaconda.org/anaconda/python>`_. 
If you for some reason do not want to use a virtual environment, you can install the dependencies for the current user only by appending 
``python3 install_dev.py --user``.

If the installer above does not work, or you want/need to install additional dependencies, 
please follow the :ref:`manual installation instructions <Introduction/pages/installation:Manual installation>` below.

PYTHONPATH
__________
To use the development version of NuRadioMC, its installation path needs to be included in the user PYTHONPATH.
If the installer does not succesfully do this, please manually add

.. code-block:: Bash

  export PYTHONPATH=/path/to/NuRadioMC:$PYTHONPATH

to your ``~/.profile``. (Depending on which terminal you use, you may have to modify one of ``.zprofile``, ``.bashrc`` or ``.zshrc`` instead).

Manual installation
-------------------

Pip-installable dependencies
____________________________

To install all (optional and non-optional) dependencies available in pip at once, use the command

.. code-block:: Bash

  pip install numpy scipy matplotlib astropy tinydb tinydb-serialization aenum h5py mysql-python pymongo dash plotly toml peakutils

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

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

These packages are recommended to be able to use all of NuRadioMC/NuRadioReco's features:

- h5py to open HDF5 files:

.. code-block:: Bash

  pip install h5py

- uproot to open RNO-G root files:

.. code-block:: bash

  pip install uproot==4.1.1

- To access some detector databases:

- For SQL datbases install `MySQL <https://www.mysql.com/>`_ and mysql-python:

  .. code-block:: Bash

    pip install mysql-python

- For `MongoDB <https://www.mongodb.com>`_ databases install:

  .. code-block:: Bash

    pip install pymongo

- To use the :ref:`Event Display <NuRadioReco/pages/event_display:Event Display>` you need plotly and dash:

  .. code-block:: Bash

    pip install dash
    pip install plotly

  If you want templates to show up in the Event Display, you need to set up an environment variable NURADIORECOTEMPLATES and have it point to the template directory.

- The documentation is created using `Sphinx <https://www.sphinx-doc.org>`_. We use the ``readthedocs`` theme, and the ``numpydoc`` format is used in our docstrings.
  This dependency is needed only if you want to generate the documentation locally - the `online documentation <https://nu-radio.github.io/NuRadioMC/main.html>`_ is generated by a Github action automatically.
  Note that we use the `sphinx autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc>`_
  feature, which tries to import all modules it documents. So if you are missing some optional dependencies, it will not generate correct documentation for all the code.

  .. code-block:: Bash

    pip install sphinx sphinx_rtd_theme numpydoc

- Some debug plots need peakutils:

  .. code-block:: Bash

    pip install peakutils

- Proposal to use :mod:`NuRadioMC.EvtGen.NuRadioProposal` module:

  .. code-block:: bash

    pip install proposal==7.5.0

  Note that the pip installation for this version of proposal may not work on all systems, in particular:

  - conda cannot be used on all systems (eg. on Mac), in that case use a python venv, see details `here <https://github.com/tudo-astroparticlephysics/PROPOSAL/issues/209>`_

  - if the linux kernel is too old (eg. on some computing clusters), refer to `this step-by-step guide <https://github.com/tudo-astroparticlephysics/PROPOSAL/wiki/Installing-PROPOSAL-on-a-Linux-kernel---4.11>`_
  

- To use the channelGalacticNoiseAdder, you need the `PyGDSM <https://github.com/telegraphic/pygdsm>`_ package.

  .. code-block:: Bash

    pip install git+https://github.com/telegraphic/pygdsm

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
  The radiopropa github, with installation instructions, can be found `here <https://github.com/nu-radio/RadioPropa>`_.
- To read ARIANNA files, `Snowshovel <https://arianna.ps.uci.edu/mediawiki/index.php/Local_DAQ_Instructions>`_ needs to be installed.
- To read ARA files, `ARA ROOT <http://www.hep.ucl.ac.uk/uhen/ara/araroot/branches/3.13/index.shtml>`_ needs to be installed.
