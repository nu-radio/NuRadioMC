import datetime

operate = datetime.datetime

def pass_configuration(station_id,evt_time):
    if station_id == 2:
       if operate(2013, 8, 26, 5,7,13) <= evt_time <= operate(2014, 4, 23, 2,13,4):
          config = 'C1'
       elif evt_time <= operate(2013,8,26,2,49,0):
            config = 'C2'
       elif  operate(2014, 4, 23, 5,22,27) <= evt_time <= operate(2014,8,6,4,21,56):
            config = 'C3'
       elif operate(2014,8,6,4,28,41) <= evt_time <= operate(2015, 11,9, 21, 54,40) or operate(2016,10,17,15,56,50) <= evt_time <= operate(2017, 11, 16, 17,40,28):
            config = 'C4'
       elif operate(2015,12,1,21,49,39) <= evt_time <= operate(2016, 10,17,12,38,18) or operate(2018, 1,1) <= evt_time <= operate(2018, 1,13):
            config = 'C5'
       elif evt_time >= operate(2018,1,14):
            config = 'C6'
       else:
            config = 'C0'
       return config

    elif station_id ==3:
       if operate(2013,10,17,22,41,23) <= evt_time <= operate(2013, 6, 11, 14,43,51) :
          config = 'C1'
       elif evt_time <= operate(2013,2,17, 22, 41, 22):
            config = 'C2'
       elif operate(2014, 8, 14, 10,24,14) <= evt_time <= operate(2015, 11,30, 10,20,10) or operate(2016,10,17,15,43,3) <= evt_time <= operate(2016, 11,16,19,45,2):
            config = 'C3'
       elif operate(2015,11,30,10,20,17) < evt_time < operate(2016, 10,17,12,37,12):
            config = 'C4'
       elif evt_time >= operate(2013, 6, 11, 14,43,53) and evt_time <= operate(2014,8,14, 10, 6, 46): 
            config = 'C5'
       elif operate(2018,1,18, 1, 44, 36) <= evt_time <= operate(2018, 12,7, 20, 17, 28):
            config = 'C6'
       elif evt_time >= operate(2018,12,7,20, 17,30):
            config = 'C7'
       else:
            config = 'C0'
       return config
    elif station_id == 4 or 5 or 100:
       config = 'C0'
       return config  
