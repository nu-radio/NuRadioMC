import pickle

def read_pickle(filename, encoding='latin1'):
    """
    Read in a pickle file and return the result
    This utility is supposed to provide compatibility for pickles created with
    different python versions. If a simple pickle.load fails, it will try to
    load the file with a specific encoding.
    
    Parameters
    ---------
    filename: string
        Name of the pickle file to be opened
    encoding: string
        Encoding to be used if the first attempt to open the pickle fails
    """
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except:
        with open(filename, 'rb') as file:
            return pickle.load(file, encoding=encoding)
