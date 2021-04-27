Installing NuRadioReco
========================

Installation using Pip
-------------------------

  The latest NuRadioReco release can be installed using pip:

    .. code-block:: Bash

      pip install NuRadioReco

  This will also install the required dependencies.

  .. Important:: If you want the current version or you want to contribute to NuRadioReco, you need to install it manually.

Manual Installation
---------------------------

  If you don't already have it installed, install `Git <https://git-scm.com/>`_.

  Then clone the NuRadioReco repository

  .. code-block:: Bash

    git clone https://github.com/nu-radio/NuRadioReco.git

  and add it to you PYTHONPATH.

Dependencies
--------------------------

  To install all (optional and non-optional) dependencies available in pip at once, use the command

  .. code-block:: Bash

    pip install numpy scipy matplotlib astropy tinydb tinydb-serialization aenum h5py mysql-python pymongo dash plotly sphinx peakutils

Core Dependencies
______________________

  NuRadioReco requires the following packages to work properly:

  - `radiotools <https://github.com/nu-radio/radiotools>`_: Can also be installed using git and needs to be added to the PYTHONPATH

  All other dependencies can be installed using pip

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
______________________

These packages are recommended to be able to use all of NuRadioReco's features:

  - h5py to open HDF5 files:

    .. code-block:: Bash

      pip install h5py

  - To access detector databases:

    - For SQL datbases install `MySQL <https://www.mysql.com/>`_ and mysql-python:

      .. code-block:: Bash

        pip install mysql-python

    - For `MongoDB <https://www.mongodb.com>`_ databases install:

      .. code-block:: Bash

        pip install pymongo

    - To use the `Event Display <./event_display.html>`_ you need plotly and dash:

      .. code-block:: Bash

        pip install dash
        pip install plotly

      If you want templates to show up in the Event Display, you need to set up an environment variable NURADIORECOTEMPLATES and have it point to the template directory.

    - The documentation is created using `Sphinx <https://www.sphinx-doc.org>`_

      .. code-block:: Bash

        pip install sphinx

    - Some debug plots need peakutils:

      .. code-block:: Bash

        pip install peakutils

    - To read ARIANNA files, `Snowshovel <https://arianna.ps.uci.edu/mediawiki/index.php/Local_DAQ_Instructions>`_ need to be installed.

    - To use the channelGalacticNoiseAdder, you need the `PyGDSM <https://github.com/telegraphic/pygdsm>`_ package.

      .. code-block:: Bash

        pip install git+https://github.com/telegraphic/pygdsm