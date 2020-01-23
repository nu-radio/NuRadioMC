import detector
import argparse
import datetime
import logging


det = detector.Detector(source='sql')
# print(det.get_antenna_deployment_time(52, 1))


parser = argparse.ArgumentParser(description='test DB detector description')
parser.add_argument('station_number', type=int,
                    help='the station number')
parser.add_argument("--time", dest='timestamp', type=str, default=None,
                    help='the time to evaluate the DB')
args = parser.parse_args()

det = detector.Detector()
if(args.timestamp is not None):
    print("update detector to time {}".format(args.timestamp))
    print(datetime.datetime.strptime(args.timestamp, "%Y-%m-%d"))
    det.update(datetime.datetime.strptime(args.timestamp, "%Y-%m-%d"))

# result = det.get_everything(args.station_number)
result = det.get_relative_positions(args.station_number)
print(result)


