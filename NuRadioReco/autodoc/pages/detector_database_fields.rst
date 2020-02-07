Properties of Detector Description
=========================================
This pages documents and defines the properties that are part of the detector decription


Antenna Table
-----------------------------
position_x: The x position of the antenna feed point relative to the station position 
position_y: The y position of the antenna feed point relative to the station position 
position_z: The z position of the antenna feed point relative to the station position 

The orientation of the antenna is described with 2 vectors. The first vector (referred to as 'orientation') is the orientation of the antenna, in case
of an LPDA the boresight direction (pointing into the main sensitivity direction) and in case of dipoles this vector is
parallel to the rod. 
The second vector describes a rotation around the first vector and needs to be perpendicular to the 'orientation' vector. 
For LPDAs, it is perpendicular to the antenna tines and points into the same direction as the connector of the create LPDAs. 
For dipoles, it can point in any direction that is perpendicular to the first vector because of the radial symmetry of dipoles. 

.. image:: orientation_sketch.png
   :width: 600
   
The vectors are each described by two angles:

orientation_theta: boresight direction (zenith angle, 0deg is the zenith, 180deg is straight down)
orientation_phi: boresight direction (azimuth angle counting from East counterclockwise)
rotation_theta: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector
rotation_phi: rotation of the antenna, is perpendicular to 'orientation', for LPDAs: vector in plane of tines pointing away from connector

deployment_time: the time of antenna deployment aka the time when the antenna depth was measured (relevant because the depth changes because of snow accumulation)
type: the type of antenna

Antenna positions can be visualized in 3D using the script `NuRadioReco/detector/visualize_detector.py my_detector.json`