How to get started
==================

.. contents::
    :local:
    :backlinks: none

New to NuRadioMC? This page provides some links to interactive ``.ipynb`` notebooks that should help you get started to run your first neutrino simulation,
and familiarize yourself with the :doc:`.nur data format </NuRadioReco/pages/event_structure>` used by NuRadioMC.
It also has some links to further documentation and examples.

Should you still have questions / issues after running through these, feel free to contact the developers via `github <https://github.com/nu-radio/NuRadioMC/issues>`_ or slack.

Installation
------------
To install NuRadioMC and NuRadioReco, you can use ``pip``:

.. code-block::

    pip install nuradiomc

This installs the latest release version of NuRadioMC.
If you want to use the development version, or intend to :doc:`contribute </Introduction/pages/contributing>`, see the installation instructions :ref:`here <Introduction/pages/installation:Development version>`.

Simulating a neutrino detector with NuRadioMC
---------------------------------------------
`Interactive notebook example showing how to simulate a neutrino detector <https://github.com/nu-radio/NuRadioMC/tree/develop/NuRadioMC/examples/Interactive/W01-simulate-neutrino-detector.ipynb>`_

This example can be found at ``NuRadioMC/examples/Interactive/W01-simulate-neutrino-detector.ipynb``.
It is based on the examples at ``NuRadioMC/examples/01_Veff_simulation`` and ``NuRadioMC/examples/07_RNO_G_simulation``.
Additional information about running a simulation can be found :doc:`here</NuRadioMC/pages/Manuals/veff_tutorial>`.

Reading a ``.nur`` file
-----------------------
`Interactive notebook example showing how to read a .nur file <https://github.com/nu-radio/NuRadioMC/tree/develop/NuRadioReco/examples/Interactive/W02-reading-nur-files.ipynb>`_

This example can be found at ``NuRadioReco/examples/Interactive/W02-reading-nur-files.ipynb``.
It shows how to navigate the quantities stored in :doc:`NuRadio's custom data structure </NuRadioReco/pages/event_structure>`, the `.nur` file.
An extensive description of the NuRadio data structure can be found :doc:`here </NuRadioReco/pages/event_structure>`.

Running a simple analysis with NuRadioReco
------------------------------------------
(WIP)
