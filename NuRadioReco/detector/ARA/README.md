#The ARA detector database json file is produced by the following script.
 - PrintJsonFile.C
This script depends on ARA environment. Exect details of setting it up may vary from system to system. When the central CVMFS installation of the ARA software is available as, for example, on Cobalt on the IceCube cluster at the Univesity of Wisconsin-Madison, one can set up the environment executing the following from the linux shell:
   source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh

NOTE: the above setup sets ROOT so that ara_rootlogon.C is loaded when a ROOT session is started. However, this could be overwritten if a user has Rint.Logon set in their .rootrc. If that is the case, one should execute the content of ara_rootlogon.C manually from the ROOT prompt before proceeding with commands below, or add its content to one's own rootlogon.C. One can check what rootlogon is actually loaded by executing from ROOT prompt the command gEnv->Print(). The full path to ARA's rootlogon is /cvmfs/ara.opensciencegrid.org/trunk/centos7/ara_build/macros/ara_rootlogon.C

#To run the script, start ROOT, and from the ROOT prompt execute
- .L PrintJsonFile.C++
 - PrintJsonFile()

#This script retrieves antenna locations from AraRoot database in station-centric coordinates and converts them to global coordinates.
- The global cordinates are computed using 126.77 (90+36.77) degree of clockwise rotation from the direction of ice flow
- The antenna types are set to point to XFDTD simulation of ARA birdcage and quad-slot cylinder antennas from Chiba (the best representation of ARA Vpol and Hpol antennas to date).

