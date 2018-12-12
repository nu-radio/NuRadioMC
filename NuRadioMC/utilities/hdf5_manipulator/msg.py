"""
Messenger for HDF5 Manipulator
"""

VIOLET = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m'


def _print(text, indent, color):

    """Print text in given color, preceded by given #tabs.

    Keyword arguments:
    text -- text to print
    indent -- no. of tabs
    color -- text color defined by global variables
    """

    if indent:
        print '\t' * indent,
    print color + text + END


def info(text, indent=None):

    """Precede text with 'INFO' and call _print with GREEN color."""

    _print("\nINFO: " + text, indent, GREEN)


def warning(text, indent=None):

    """Precede text with 'WARNING' and call _print with YELLOW color."""

    _print("\nWARNING: " + text, indent, YELLOW)


def error(text, indent=None):

    """Precede text with 'ERROR' and call _print with RED color."""

    _print("\nERROR: " + text, indent, RED)


def box(text, width=80):

    """'Draw box' and print text in the center (using BOLD font)."""

    print BLUE
    pad = (width - len(text)) / 2
    print '+' + '-' * width + '+'
    print '|' + ' ' * width + '|'
    print '|' + ' ' * pad + text + ' ' * (width - pad - len(text)) + '|'
    print '|' + ' ' * width + '|'
    print '+' + '-' * width + '+'
    print END


def list_dataset(data, indent=1):

    """Print the list of datasets.

    Keyword arguments:
    data -- dictionary with data
    """

    adjust = len(max(data.keys(), key=len)) + 1  # length of left text column

    for key in data:
        print '\t' * indent + " - %(key)s %(type)s %(size)s" \
            % {"key": (key+':').ljust(adjust),
               "size": '-> ' + str(data[key].shape),
               "type": ('[' + str(data[key].dtype) + ']').ljust(9),
               }


def list_fileinfo(filename, range):

    """Print information about file to be saved.

    Keyword arguments:
    filename -- path to output hdf5 file
    range -- subset to be saved
    """

    print "\t - %(file)s: %(n)d entries from %(b)d to %(e)d" \
          % {"file": filename, "n": range[1] - range[0],
             "b": range[0], "e": range[1] - 1}
