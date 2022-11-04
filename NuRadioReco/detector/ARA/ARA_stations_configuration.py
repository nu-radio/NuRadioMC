import sys
import datetime
import logging
logger = logging.getLogger("analog_components")

tkeeper = datetime.datetime
distant_past = tkeeper(2010,1,1)
distant_future = tkeeper(2100,1,1)
#Set AraSim gain measurement as default ARA_hardware_configuration 
ARA_hardware_config_default = 'C0'

###  we set lower and upper time boundaries of given ARA_hardware_configuration for each station in following steps  ###
#For station ARA1
boundaries_ARA1= {}
boundaries_ARA1[ARA_hardware_config_default] = [distant_past, distant_future, 'C0']
#For station ARA2
boundaries_ARA2 = {}
boundaries_ARA2['C1'] = [tkeeper(2013, 8, 26, 5, 7,13),  tkeeper(2014, 4, 23, 2,13, 4)  ,'C1']
boundaries_ARA2['C2'] = [distant_past, tkeeper(2013, 8, 26, 2,49, 0)  ,'C2']    
boundaries_ARA2['C3'] = [tkeeper(2014, 4, 23, 5,22,27), tkeeper(2014, 8,  6, 4,21,56), 'C3'] 
boundaries_ARA2['C4_range_1'] = [tkeeper(2014, 8,  6, 4,28,41), tkeeper(2015, 11,9, 21, 54,40) , 'C4'] 
boundaries_ARA2['C4_range_2'] = [tkeeper(2016,10, 17,15,56,50), tkeeper(2017, 11, 16, 17,40,28), 'C4'] 
boundaries_ARA2['C5_range_1'] = [tkeeper(2015,12,1,21,49,39), tkeeper(2016, 10,17,12,38,18), 'C5'] 
boundaries_ARA2['C5_range_2'] = [tkeeper(2018, 1,1), tkeeper(2018, 1,13), 'C5'] 
boundaries_ARA2['C6'] = [tkeeper(2018,1,14), distant_future, 'C6'] 
#For station ARA3
boundaries_ARA3 = {}
boundaries_ARA3['C1'] = [tkeeper(2013,10,17,22,41,23), tkeeper(2013, 6, 11, 14,43,51), 'C1']
boundaries_ARA3['C2'] = [distant_past, tkeeper(2013,2,17, 22, 41, 22) , 'C2']
boundaries_ARA3['C3_range_1'] = [tkeeper(2014, 8, 14, 10,24,14), tkeeper(2015, 11,30, 10,20,10), 'C3']
boundaries_ARA3['C3_range_2'] = [tkeeper(2016,10,17,15,43,3), tkeeper(2016, 11,16,19,45,2), 'C3']
boundaries_ARA3['C4'] = [tkeeper(2015,11,30,10,20,17), tkeeper(2016, 10,17,12,37,12), 'C4']
boundaries_ARA3['C5'] = [tkeeper(2013, 6, 11, 14,43,53), tkeeper(2014,8,14, 10, 6, 46), 'C5']
boundaries_ARA3['C6'] = [tkeeper(2018,1,18, 1, 44, 36), tkeeper(2018, 12,7, 20, 17, 28), 'C6']
boundaries_ARA3['C7'] = [tkeeper(2018,12,7,20, 17,30), distant_future, 'C7']
#For station ARA4
boundaries_ARA4 = {}
boundaries_ARA4[ARA_hardware_config_default] = [distant_past, distant_future, ARA_hardware_config_default]
#For station ARA5
boundaries_ARA5 = {}
boundaries_ARA5[ARA_hardware_config_default] = [distant_past, distant_future, ARA_hardware_config_default]

boundaries_all = {}
boundaries_all[100] = boundaries_ARA1
boundaries_all[2] = boundaries_ARA2
boundaries_all[3] = boundaries_ARA3
boundaries_all[4] = boundaries_ARA4
boundaries_all[5] = boundaries_ARA5

def get_ARA_hardware_configuration(station_id,evt_time):
    if station_id not in boundaries_all:
       logger.error(" Station ID {} is unknown, please make sure to use valid station id in your detector description file".format(station_id))
       sys.exit(-1) 
       #return ARA_hardware_config_default
    get_boundaries = boundaries_all[ station_id ]
    ARA_hardware_config = ARA_hardware_config_default
    for period in get_boundaries:
        if get_boundaries[period][0] <= evt_time <= get_boundaries[period][1]:
            ARA_hardware_config = get_boundaries[period][2]
            break
    return ARA_hardware_config







