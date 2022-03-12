import NuRadioReco.detector.detector_mongo
import argparse
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test Mongo DB detector description')
    parser.add_argument("--time", dest='timestamp', type=str, default=None,
                        help='the time to evaluate the DB')
    args = parser.parse_args()

    det = NuRadioReco.detector.detector_mongo.Detector()
    if args.timestamp is not None:
        det_time = datetime.datetime.strptime(args.timestamp, "%Y-%m-%d")
    else:
        det_time = datetime.datetime.now()
    print("update detector to time {}".format(det_time))
    det.update(det_time)


    time_str = det_time.strftime("%Y_%m_%d")
    det.export_detector(f"mongodb_{time_str}.json")    
