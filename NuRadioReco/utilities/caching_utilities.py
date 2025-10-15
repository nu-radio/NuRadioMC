"""
Caching utilities for NuRadioReco/NuRadioMC

This module provides general-purpose caching functions that can be used by various
modules to cache computed data to disk and avoid repeated expensive calculations.

Example usage:
    
    # Generate cache key from configuration
    cache_key = generate_cache_key(station_id, channels, limits, step_sizes)
    
    # Get cache file path
    cache_path = get_cache_path("delay_matrices", f"station{station_id}_{cache_key}.pkl")
    
    # Try to load from cache
    cached_data = load_from_cache(cache_path)
    if cached_data is None:
        # Compute data...
        data = compute_expensive_data()
        save_to_cache(data, cache_path, metadata={'station': station_id})
    else:
        data = cached_data
"""

import os
import pickle
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger('NuRadioReco.utilities.caching_utilities')


def generate_cache_key(*args, **kwargs):
    """
    Generate a unique MD5 hash key from arbitrary arguments.
    
    This function creates a deterministic hash from any combination of positional
    and keyword arguments. Useful for creating cache keys based on configuration
    parameters.
    
    Parameters
    ----------
    *args : tuple
        Positional arguments to include in hash
    **kwargs : dict
        Keyword arguments to include in hash
        
    Returns
    -------
    str
        16-character hexadecimal hash string
        
    Examples
    --------
    >>> key1 = generate_cache_key(21, [0,1,2,3], coord_system='cylindrical')
    >>> key2 = generate_cache_key(21, [0,1,2,3], coord_system='cylindrical')
    >>> key1 == key2
    True
    """
    # Convert to string representation and sort kwargs for deterministic hashing
    config_str = str((args, sorted(kwargs.items())))
    cache_key = hashlib.md5(config_str.encode()).hexdigest()[:16]
    return cache_key


def get_cache_path(cache_subdir, filename, cache_root="~/.cache/nuradio"):
    """
    Get the full path to a cache file, creating directories if needed.
    
    Creates an organized cache directory structure under the cache root.
    Falls back to /tmp if the cache_root is not writable.
    
    Parameters
    ----------
    cache_subdir : str
        Subdirectory under cache root (e.g., 'delay_matrices', 'antenna_patterns')
    filename : str
        Name of the cache file
    cache_root : str, optional
        Root directory for all caching (default: '~/.cache/nuradio')
        
    Returns
    -------
    str
        Full path to cache file
        
    Examples
    --------
    >>> path = get_cache_path('delay_matrices', 'station21_abc123.pkl')
    >>> path
    '/home/user/.cache/nuradio/delay_matrices/station21_abc123.pkl'
    """
    cache_dir = os.path.expanduser(os.path.join(cache_root, cache_subdir))
    
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except (OSError, PermissionError):
        # Fall back to temp directory if home cache is not writable
        cache_dir = f"/tmp/nuradio_{cache_subdir}_{os.getuid()}"
        os.makedirs(cache_dir, exist_ok=True)
        logger.warning(f"Could not create cache in {cache_root}, using {cache_dir}")
    
    return os.path.join(cache_dir, filename)


def load_from_cache(filepath):
    """
    Load cached data from a pickle file.
    
    Parameters
    ----------
    filepath : str
        Full path to cache file
        
    Returns
    -------
    data : object or None
        The cached data, or None if file doesn't exist or loading fails
        
    Examples
    --------
    >>> data = load_from_cache('/path/to/cache.pkl')
    >>> if data is not None:
    ...     print("Loaded from cache!")
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)
        
        logger.debug(f"Loaded data from cache: {filepath}")
        
        # If the cache was saved with metadata wrapper, extract the data
        if isinstance(cache_data, dict) and 'data' in cache_data:
            return cache_data['data']
        else:
            # Backward compatibility: return the raw data
            return cache_data
            
    except Exception as e:
        logger.warning(f"Failed to load cache from {filepath}: {e}")
        return None


def save_to_cache(data, filepath, metadata=None):
    """
    Save data to cache with optional metadata.
    
    The data is wrapped in a dictionary with timestamp and optional metadata
    before being pickled. This helps with cache management and debugging.
    
    Parameters
    ----------
    data : object
        The data to cache (must be pickle-able)
    filepath : str
        Full path where cache file should be saved
    metadata : dict, optional
        Additional metadata to store with the cached data
        
    Examples
    --------
    >>> save_to_cache(delay_matrices, cache_path, 
    ...               metadata={'station': 21, 'channels': [0,1,2,3]})
    """
    cache_data = {
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved data to cache: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to save cache to {filepath}: {e}")


def clear_cache(cache_subdir=None, cache_root="~/.cache/nuradio"):
    """
    Clear cached files from a specific subdirectory or entire cache.
    
    Parameters
    ----------
    cache_subdir : str, optional
        Subdirectory to clear. If None, clears entire cache root.
    cache_root : str, optional
        Root directory for caching (default: '~/.cache/nuradio')
        
    Returns
    -------
    int
        Number of files deleted
        
    Examples
    --------
    >>> # Clear all delay matrix caches
    >>> n_deleted = clear_cache('delay_matrices')
    >>> print(f"Deleted {n_deleted} cache files")
    """
    import glob
    
    if cache_subdir is not None:
        cache_dir = os.path.expanduser(os.path.join(cache_root, cache_subdir))
    else:
        cache_dir = os.path.expanduser(cache_root)
    
    if not os.path.exists(cache_dir):
        logger.info(f"Cache directory does not exist: {cache_dir}")
        return 0
    
    cache_files = glob.glob(os.path.join(cache_dir, "*.pkl"))
    n_deleted = 0
    
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            n_deleted += 1
        except Exception as e:
            logger.warning(f"Failed to delete {cache_file}: {e}")
    
    logger.info(f"Deleted {n_deleted} cache files from {cache_dir}")
    return n_deleted


def get_cache_info(cache_subdir, cache_root="~/.cache/nuradio"):
    """
    Get information about cached files in a subdirectory.
    
    Parameters
    ----------
    cache_subdir : str
        Subdirectory to inspect
    cache_root : str, optional
        Root directory for caching (default: '~/.cache/nuradio')
        
    Returns
    -------
    list of dict
        List of dictionaries with cache file information (path, size, timestamp)
        
    Examples
    --------
    >>> info = get_cache_info('delay_matrices')
    >>> for item in info:
    ...     print(f"{item['path']}: {item['size']/1e6:.1f} MB")
    """
    import glob
    
    cache_dir = os.path.expanduser(os.path.join(cache_root, cache_subdir))
    
    if not os.path.exists(cache_dir):
        return []
    
    cache_files = glob.glob(os.path.join(cache_dir, "*.pkl"))
    info_list = []
    
    for cache_file in cache_files:
        try:
            stat = os.stat(cache_file)
            
            # Try to load metadata
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            if isinstance(cache_data, dict) and 'timestamp' in cache_data:
                timestamp = cache_data['timestamp']
                metadata = cache_data.get('metadata', {})
            else:
                timestamp = datetime.fromtimestamp(stat.st_mtime).isoformat()
                metadata = {}
            
            info_list.append({
                'path': cache_file,
                'filename': os.path.basename(cache_file),
                'size': stat.st_size,
                'timestamp': timestamp,
                'metadata': metadata
            })
        except Exception as e:
            logger.warning(f"Failed to get info for {cache_file}: {e}")
    
    return info_list
