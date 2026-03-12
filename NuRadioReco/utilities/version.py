import NuRadioMC
import NuRadioReco

from subprocess import Popen, PIPE
import os
import re
import logging
logger = logging.getLogger("NuRadioReco.utilities.version")


def get_git_commit_hash(module) -> str:
    try:
        if not module.__version__.endswith('-dev'):
            # distribution versions won't have a Git repository;
            # eagerly return to not waste time
            return "none"

        path = os.path.dirname(module.__file__)

        gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE, cwd=path)
        (stdout, stderr) = gitproc.communicate()
        h = stdout.decode('utf-8').strip()
        check = re.compile(r"^[a-f0-9]{40}(:.+)?$", re.IGNORECASE)
        assert check.match(h)
    except:
        logger.error("NuRadioMC/Reco version could not be determined, returning None")
        return "none"
    return h


# preload in case the source changes during execution
_NuRadioMC_hash = get_git_commit_hash(NuRadioMC)
_NuRadioReco_hash = get_git_commit_hash(NuRadioReco)


def get_NuRadioMC_commit_hash() -> str:
    """
    returns the hash of the current commit of the NuRadioMC git repository
    """
    return _NuRadioMC_hash


def get_NuRadioReco_commit_hash() -> str:
    """
    returns the hash of the current commit of the NuRadioReco git repository
    """
    return _NuRadioReco_hash
