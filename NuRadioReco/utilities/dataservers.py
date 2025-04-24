import requests
import os
import filelock
import logging
import shutil
from glob import glob

logger = logging.getLogger('NuRadioReco.dataservers')

dataservers = ["https://rnog-data.zeuthen.desy.de", "https://rno-g.uchicago.edu/data/desy-mirror"]

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
        except Exception as _:
            continue

        response_times.append(response.elapsed)
        available_dataservers.append(dataserver)

    ranked_dataservers = [x for _, x in sorted(zip(response_times, available_dataservers))]
    return ranked_dataservers


def download_from_dataserver(remote_path, target_path, unpack_tarball=True, dataservers=dataservers, try_ordered=False):
    """ download remote_path to target_path from the list of NuRadio dataservers """

    folder = os.path.dirname(target_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    lockfile = target_path+".lock"
    lock = filelock.FileLock(lockfile)

    logger.warning(f"Assuring no other process is downloading. Will wait until {lockfile} is unlocked.")
    with lock:
        if os.path.isfile(target_path):
            logger.warning(f"{target_path} already exists. Maybe download was already completed by another instance?")
            return
        elif unpack_tarball and (len(glob(os.path.dirname(target_path) + "/*.dat")) > 0): #just check if any .dat files present (similar to NuRadioProposal.py)
            logger.warning(f"{os.path.dirname(target_path)} contains .dat files. Maybe download was already completed by another instance?")
            return

        if try_ordered:
            dataservers = get_available_dataservers_by_responsetime(dataservers)

        requests_status = requests.codes["not_found"]
        for dataserver in dataservers:
            URL = f'{dataserver}/{remote_path}'

            logger.warning(
                "downloading file {} from {}. This can take a while...".format(target_path, URL))

            try:
                r = requests.get(URL)
                r.raise_for_status()
                requests_status = r.status_code
                break
            except requests.exceptions.HTTPError:
                logger.warning(f"HTTP Error for {dataserver}. Does the file {remote_path} exist on the server?")
                pass
            except requests.exceptions.ConnectionError:
                logger.warning(f"Error Connecting to {dataserver}. Maybe you don't have internet... or the server is down?")
                pass
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout Error for {dataserver}.")
                pass
            except requests.exceptions.RequestException as err:
                logger.warning(f"An unusual error for {dataserver} occurred: {err}")
                pass

            logger.warning("problem downloading file {} from {}. Let's see if there is another server.".format(target_path, URL))

        if requests_status != requests.codes["ok"]:
            logger.error(f"error in download of file {target_path}. Tried all servers in {dataservers} without success.")
            raise IOError

        with open(target_path, "wb") as code:
            code.write(r.content)
        logger.warning("...download finished.")

        if unpack_tarball and target_path.endswith(".tar.gz"):
            target_dir = os.path.dirname(target_path)
            logger.warning(f"...unpacking archive to {target_dir}")
            shutil.unpack_archive(target_path, target_dir)
            os.remove(target_path)
