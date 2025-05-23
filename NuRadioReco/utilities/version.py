import NuRadioMC
import NuRadioReco

from subprocess import Popen, PIPE
import os
import re
import logging
logger = logging.getLogger("NuRadioReco.utilities.version")


def get_git_commit_hash(path):
    try:
        gitproc = Popen(['git', 'show'], stdout=PIPE, cwd=path)
        (stdout, stderr) = gitproc.communicate()
        h = stdout.decode('utf-8').split('\n')[0].split()[1]
        check = re.compile(r"^[a-f0-9]{40}(:.+)?$", re.IGNORECASE)
        if(not check.match(h)):
            logger.error("NuRadioMC version could not be determined, returning None")
            return "none"
    except:
        return "none"
    return h

path = os.path.dirname(NuRadioMC.__file__)
NuRadioMC_hash = get_git_commit_hash(path)


def get_NuRadioMC_commit_hash():
    """
    returns the hash of the current commit of the NuRadioMC git repository
    """
    return NuRadioMC_hash

path = os.path.dirname(NuRadioReco.__file__)
NuRadioReco_hash = get_git_commit_hash(path)

def get_NuRadioReco_commit_hash():
    """
    returns the hash of the current commit of the NuRadioReco git repository
    """
    return NuRadioReco_hash
