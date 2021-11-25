import numpy as np

site = 2
## site 1: station 21, site 2: station 11, site 3: station 22
## csv file with data can be found here: https://radio.uchicago.edu/wiki/index.php/Deployment
## Station positions are with respect the GPS base station on the MSF roof
data = np.genfromtxt(
        'data/survey_results.csv',
        delimiter=',',
        skip_header=5,
        dtype=("|S20", float, float, float, float, float, "|S10")
    )
    
data_site = []
for i_row, row in enumerate(data):
    if 'site {} power'.format(site) in row[0].decode('UTF-8'):
        power_easting = row[1]
        power_northing = row[2]
        print("Determine relative positions for strings and surface antennas for site {}:".format(site))
        print("position of power string easting: {}, northing: {}".format(power_easting, power_northing))
        print("altitude of power string:", row[3])
        print("_______________________________________________________")
for i_row, row in enumerate(data):
    if 'site {}'.format(site) in row[0].decode('UTF-8'):
        easting = row[1]
        northing = row[2]
        print("{}| \n rel position easting: {}, rel position northing: {} \n distance to power string: {}".format(row[0].decode('UTF-8'), easting - power_easting, northing - power_northing, np.sqrt((easting - power_easting)**2 + (northing - power_northing)**2)))

    
