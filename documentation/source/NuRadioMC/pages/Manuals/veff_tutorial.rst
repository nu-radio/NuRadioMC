Calculating effective volume
============================

This tutorial gives a full example to simulate the effective volume of a high-energy radio neutrino detector.

Installation
------------
See :doc:`here </Introduction/pages/installation>` for installation instructions.

Run an effective volume simulation
----------------------------------
An effective volume simulation consists of the following steps

* generation of neutrino interactions in a cylindrical volume
* calculating the Askaryan signal (ray tracing)
* propagating the radio signal to your antennas
* accounting for signal attenuation
* simulating the antenna response
* simulating the trigger

A working example is provided in ``NuRadioMC/examples/01_Veff_simulation``. Do NOT execute this example within this directory but copy it to a separate place outside of the NuRadioMC github repository. E.g. 

.. code-block:: bash

    mkdir $HOME/simulations
    cp -r $HOME/software/NuRadioMC/NuRadioMC/examples/01_Veff_simulation $HOME/simulations/
    cd $HOME/simulations/01_Veff_simulation


To run a simulation you need

* an input event list specifying the neutrino vertices, energy, direction, flavor, type of interaction (charged current or neutral current), etc. 
* a detector description defining the positions of your antennas, the antenna type and orientation, sampling frequency, etc. 
* a NuRadioMC config file that e.g. specifies which Askaryan module and ice model to use
* a main run script that also defines the details of the detector simulation

Generating the input event list
-------------------------------
Generating the input event list is easy using the NuRadioMC event generator. Just execute

.. code-block:: bash
    
    python T01generate_event_list.py

The script will generate two event lists. One with 1000 events at 1e19 eV neutrino energy and one with 10,000 events at 1e18 eV neutrino energy.

Running the simulation
-------------------------
To run the simulation execute 

.. code-block:: bash

    python T02RunSimulation.py 1e19_n1e3.hdf5 surface_station_1GHz.json config.yaml 1e19_n1e3_output.hdf5

The simulation only takes a few seconds (with the _C_ ray tracing implementation installed). The final output should be something like (small differences in the number of triggered events are expected because of random differences in the input data set)

.. code-block:: sh
    
    WARNING:sim:fraction of triggered events = 18/1000 = 0.018
    WARNING:sim:Veff = 22 km^3 sr
    WARNING:sim:1000 events processed in 35 seconds = 35.26ms/event

and all triggered events are saved in the ``1e19_n1e3_output.hdf5`` hdf5 output file. In this example only three events triggered. 
The output file only contains meta information such as the incoming signal direction, the ray tracing solutions, the polarization etc. 
If you also want to save the pulse forms just add another command line argument.

.. code-block:: bash

    python T02RunSimulation.py 1e19_n1e3.hdf5 surface_station_1GHz.json config.yaml 1e19_n1e3_output.hdf5 1e19_n1e3_output.nur

The waveforms are saved in a custom binary format that serializes the complete NuRadioReco event structure into a file. This has the advantage that you can read it into NuRadioReco again to e.g. perform a reconstruction on the simulated data. 

You can run the same simulation on the other input file with 10,000 events 

.. code-block:: bash

    python T02RunSimulation.py 1e18_n1e4.hdf5 surface_station_1GHz.json config.yaml 1e18_n1e4_output.hdf5

which takes 71 seconds on my laptop and leads to 29 triggered events.

More details: the detector description
--------------------------------------
The detector is defined in a JSON file and allows you to specify every detail of your detector that can have a relevance for the simulation or later reconstruction. 
For our simple example though, it contains many parameters that we don't need to worry about, e.g. details about the ADC, so just ignore those fields. 
The file ``surface_station_1GHz.json`` defines a 'surface station' consisting of 4 downward pointing LPDAs at -2m depth and 4 bicone antennas at -5m depth.

More info about detector files can be found :doc:`here</NuRadioReco/pages/detector/detector>`.

More details: the config file
--------------------------------
An overview of all parameters can be found in the default config file `config_default <https://github.com/nu-radio/NuRadioMC/blob/master/NuRadioMC/simulation/config_default.yaml>`_. 
Everything defined in the local configuration file ``config.yaml`` will override the default parameters. The config file uses the YAML format, an easy to use and human readable format. 
It is similar to JSON but easier to type down. 

Visualization of results
------------------------
NuRadioMC includes visualization tools. To produce the typical debugging plots of an effective volume simulation execute

.. code-block:: bash
    
    python $HOME/software/NuRadioMC/NuRadioMC/simulation/scripts/T05visualize_sim_output.py 1e18_n1e4_output.hdf5

Please note that the number of triggered events is so small that some of the plots won't make a lot of sense. 


