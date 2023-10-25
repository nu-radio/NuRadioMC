import threading
import logging
import six
import NuRadioReco.utilities.metaclasses
import os
import time
from NuRadioReco.modules.io import NuRadioRecoio

logging.basicConfig()
logger = logging.getLogger('eventbrowser.dataprovider')
logger.setLevel(logging.INFO)

try:
    from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import _readRNOGData_eventbrowser
except ImportError as e:
    logger.error(
        msg=(
            "Failed to import NuRadioReco.modules.io.RNO_G.readRNOGDataMattak, `.root` files can not be read."
            " If you are only trying to read .nur files this error can be ignored.."
        ), exc_info=e
    )
    _readRNOGData_eventbrowser = None # if we don't define this we'll raise more errors later

@six.add_metaclass(NuRadioReco.utilities.metaclasses.Singleton)
class DataProvider(object):
    _use_root = None

    def __init__(self, use_root=False, max_user_instances=12):
        """"
        Interface to .nur or .root file IO for the eventbrowser

        Parameters
        ----------
        use_root: bool, default False
            If True, use the the RNO-G specific root file reader. Otherwise,
            use the NuRadioMC
        max_user_instances: int, default=12
            Each unique session id gets its own reader, up to a maximum
            of ``max_user_instances`` concurrent readers. Subsequent
            requests for new readers drop older readers.

        """
        logger.info("Creating new DataProvider instance")
        self.__max_user_instances = max_user_instances
        self.__user_instances = {}
        self.__lock = threading.Lock()
        self.set_filetype(use_root=use_root)

    def set_filetype(self, use_root):
        """
        Set the filetype to read in.

        Parameters
        ----------
        use_root: bool
            If True, use the :mod:`NuRadioReco.modules.io.RNO_G.readRNOGDataMattak` module
            to read in RNO-G ROOT files. Otherwise, use the 'standard' NuRadioMC `.nur` file
            reader.
        """
        if self._use_root == use_root:
            return None
        if use_root:
            if _readRNOGData_eventbrowser is None:
                raise ImportError(
                    "The .root reading interface `NuRadioReco.modules.io.RNO_G.readRNOGDataMattak`"
                    " is not available, so .root files can not be read. Make sure you have a working installation"
                    " of mattak (https://github.com/RNO-G/mattak)."
                )
            self.__file_handler = _readRNOGData_eventbrowser # we only initialize once, maybe this saves some time?
        else:
            self.__file_handler = NuRadioRecoio.NuRadioRecoio # need to initialize for every file
        self._use_root = use_root

    def get_file_handler(self, user_id, filename):
        """
        Interface to retrieve the actual IO module

        Thread-locked to avoid competing threads reading the same file.

        Parameters
        ----------
        user_id: str
            unique user_id to allow multiple users to use the dataprovider at once
        filename: str | list
            path or paths to files to read in

        """
        # because the dash app is multi-threaded, we use a lock to avoid
        # multiple simultaneous requests to read the same file

        thread = threading.get_ident()
        total_threads = threading.active_count()
        logger.debug(f"Thread {thread} out of total {total_threads} requesting file_handler for user {user_id}")
        with self.__lock:
            logger.debug(f"Thread {thread} locked, getting file_handler for user {user_id}...")

            if filename is None:
                return None
            if user_id not in self.__user_instances:
                reader = self.__file_handler
                self.__user_instances[user_id] = dict(
                    reader=reader, filename=None,
                )
            if isinstance(filename, str):
                filename = [filename]
            if filename != self.__user_instances[user_id]['filename']:
                # user is requesting new file -> close current file and open new one
                reader = self.__user_instances[user_id]['reader']
                if self._use_root:
                    reader = self.__file_handler(load_run_table=False)
                    reader.begin([os.path.dirname(f) for f in filename], overwrite_sampling_rate=3.2)
                    reader.get_event_ids()
                else:
                    reader = self.__file_handler(filename) # NuRadioRecoio takes filenames as argument to __init__
                self.__user_instances[user_id] = dict(
                    reader=reader, filename=filename,
                )

            # update last access time
            self.__user_instances[user_id]['last_access_time'] = time.time()

            # check if we exceed maximum number of concurrent sessions
            if len(self.__user_instances) > self.__max_user_instances:
                users = {
                    self.__user_instances[k]['last_access_time']:k
                    for k in self.__user_instances.keys()
                }
                users_by_access_time = sorted(users)
                # drop oldest session
                self.__user_instances.pop(users_by_access_time[0])

        logger.debug(f"Returning file_handler and releasing lock")
        return self.__user_instances[user_id]['reader']
