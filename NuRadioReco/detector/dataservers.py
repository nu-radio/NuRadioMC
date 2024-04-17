import requests
import os

import socket
import pytz
from datetime import datetime
from geolite2 import geolite2

import logging
logger = logging.getLogger('NuRadioReco.dataservers')

dataservers = ["https://rnog-data.zeuthen.desy.de", "https://rno-g.uchicago.edu/data/desy-mirror", "https://rno-g.tash.cb"]

def get_available_dataservers_by_responsetime(dataservers=dataservers):
    """ requests a small index file from the list of dataservers and returns a list of responsive ones ordered by elapsed time """
    response_times = []
    available_dataservers = []

    for dataserver in dataservers:
        # get the index of the shower_library directory, because it is short
        testdir = f"{dataserver}/shower_library/"
        try:
            response = requests.get(testdir, timeout=5)
            response.raise_for_status()
        except:
            continue
        response_times.append(response.elapsed)
        available_dataservers.append(dataserver)
    ranked_dataservers = [x for _, x in sorted(zip(response_times, available_dataservers))]
    return ranked_dataservers

def get_available_dataservers_by_timezone(dataservers=dataservers):
    """ uses the server locations' timezones from the list of dataservers and returns the list of dataservers ordered by proximity """
    geo = geolite2.reader()

    naive = datetime.utcnow()
    utcoffset_local = naive.astimezone().utcoffset().total_seconds()/3600
    server_offsets = []
    for dataserver in dataservers:
        dataserver_ip = socket.gethostbyname(dataserver)
        dataserver_timezone = geo.get(dataserver_ip)["location"]["time_zone"]
        timezone = pytz.timezone(dataserver_timezone)
        utcoffset_server = timezone.localize(naive).utcoffset().total_seconds()/3600

        server_offsets.append((utcoffset_local-utcoffset_server)%12)

    ranked_dataservers = [x for _, x in sorted(zip(server_offsets, dataservers))]
    return ranked_dataservers

def download_from_dataserver(remote_path, target_path, dataservers=dataservers, try_ordered=False):
    if try_ordered:
        dataservers = get_available_dataservers_by_timezone(dataservers)

    requests_status = requests.codes["not_found"]
    for dataserver in dataservers:
        URL = f'{dataserver}/{remote_path}'

        logger.info(
            "downloading file {} from {}. This can take a while...".format(target_path, URL))

        try:
            r = requests.get(URL)
            r.raise_for_status()
            requests_status = r.status_code
            break
        except requests.exceptions.HTTPError as errh:
            logger.info(f"HTTP Error for {dataserver}:",errh)
            pass
        except requests.exceptions.ConnectionError as errc:
            logger.info(f"Error Connecting to {dataserver}:",errc)
            pass
        except requests.exceptions.Timeout as errt:
            logger.info(f"Timeout Error for {dataserver}:",errt)
            pass
        except requests.exceptions.RequestException as err:
            logger.info(f"Another Error",err)
            pass
            
        logger.warning("problem downloading file {} from {}. Let's see if there is another server.".format(target_path, URL))

    if requests_status != requests.codes["ok"]:
        logger.error(f"error in download of file {target_path}. Tried all servers in {dataservers} without success.")
        raise IOError

    folder = os.path.dirname(target_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(target_path, "wb") as code:
        code.write(r.content)
    logger.warning("...download finished.")
