"""
I/O utilities for interferometric reconstruction

This module provides functions for saving/loading interferometric reconstruction results
in various formats (HDF5, NUR, pickle), as well as utilities for organizing file paths
and extracting metadata from file names.
"""

import os
import re
import pickle
import logging
import numpy as np

logger = logging.getLogger('NuRadioReco.utilities.interferometry_io_utilities')


def extract_station_run_from_path(file_path):
    """
    Extract station and run information from correlation map file path.
    
    Parameters
    ----------
    file_path : str
        Path to correlation map pickle file
    
    Returns
    -------
    tuple
        (station_id, run_number) or (None, None) if not found
    """
    filename = os.path.basename(file_path)
    
    match = re.search(r'station(\d+)_run(\d+)_evt\d+_corrmap\.pkl', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    path_parts = file_path.split(os.sep)
    station_id = None
    run_number = None
    
    for i, part in enumerate(path_parts):
        station_match = re.match(r'station(\d+)', part)
        if station_match:
            station_id = int(station_match.group(1))
        
        run_match = re.match(r'run(\d+)', part)
        if run_match:
            run_number = int(run_match.group(1))
    
    return station_id, run_number


def create_organized_paths(config, run_number, output_type):
    """
    Create organized directory structure and file paths for results and maps.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    station_id : int
        Station ID
    run_number : int
        Run number
    output_type : str
        Output file type ('hdf5' or 'nur')
    
    Returns
    -------
    tuple : (results_path, maps_dir)
        Full path to results file and directory for correlation maps
    """
    results_base = config.get('save_results_to', './results/')
    station_id = config.get('station_id')
    
    station_dir = os.path.join(results_base, f"station{station_id}", f"run{run_number}")
    
    reco_data_dir = os.path.join(station_dir, "reco_data")
    corr_map_dir = os.path.join(station_dir, "corr_map_data")
    
    os.makedirs(reco_data_dir, exist_ok=True)
    os.makedirs(corr_map_dir, exist_ok=True)
    
    extension = 'h5' if output_type == 'hdf5' else 'nur'
    results_filename = f"station{station_id}_run{run_number}_reco_results.{extension}"
    results_path = os.path.join(reco_data_dir, results_filename)
    
    return results_path, corr_map_dir


def determine_plot_output_path(file_path, output_arg, station_id, run_number, event_number):
    """
    Determine the output path for correlation map plots following the organized structure.
    
    Parameters
    ----------
    file_path : str
        Input correlation map file path
    output_arg : str or None
        User-provided output argument
    station_id : int
        Station ID
    run_number : int
        Run number
    event_number : int
        Event number
    
    Returns
    -------
    str
        Full path for output plot file
    """
    plot_filename = f"station{station_id}_run{run_number}_evt{event_number}_corrmap.png"
    
    if output_arg is None:
        output_dir = os.path.join("figures", f"station{station_id}", f"run{run_number}")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, plot_filename)
    
    elif os.path.isdir(output_arg) or output_arg.endswith('/'):
        base_dir = output_arg.rstrip('/')
        output_dir = os.path.join(base_dir, "figures", f"station{station_id}", f"run{run_number}")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, plot_filename)
    
    else:
        output_dir = os.path.dirname(output_arg)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return output_arg


def save_interferometric_results_hdf5(results, filepath, config):
    """
    Save interferometric reconstruction results to HDF5 format.
    
    Parameters
    ----------
    results : list of dict
        List of result dictionaries, one per event
    filepath : str
        Path to output HDF5 file
    config : dict
        Configuration dictionary containing reconstruction parameters
    """
    import h5py
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        config_group = f.create_group('config')
        for key, value in config.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                config_group.attrs[key] = np.array(value)
            else:
                config_group.attrs[key] = value
        
        results_group = f.create_group('results')
        
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        for key in all_keys:
            values = []
            for result in results:
                if key in result:
                    values.append(result[key])
                else:
                    values.append(None)
            
            if all(v is None for v in values):
                continue
            
            if key in ['station_id', 'run_number', 'event_number']:
                data = np.array([v if v is not None else -1 for v in values], dtype=int)
                results_group.create_dataset(key, data=data)
            
            elif key in ['zenith', 'azimuth', 'zenith_alt', 'azimuth_alt', 
                        'correlation_value', 'correlation_value_alt']:
                data = np.array([v if v is not None else np.nan for v in values], dtype=float)
                results_group.create_dataset(key, data=data)
            
            elif key in ['max_indices', 'alt_indices']:
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    data = np.array([v if v is not None else (-1, -1) for v in values], dtype=int)
                    results_group.create_dataset(key, data=data)
            
            else:
                try:
                    data = np.array(values)
                    results_group.create_dataset(key, data=data)
                except (ValueError, TypeError):
                    logger.warning(f"Could not save field '{key}' to HDF5 - unsupported type")
    
    logger.info(f"Saved {len(results)} reconstruction results to {filepath}")


def save_interferometric_results_nur(events, filepath):
    """
    Save interferometric reconstruction results to NUR format.
    
    Parameters
    ----------
    events : list
        List of NuRadio Event objects with reconstruction parameters stored
    filepath : str
        Path to output NUR file
    """
    from NuRadioReco.framework.event import Event
    from NuRadioReco.modules.io.NuRadioRecoio import NuRadioRecoio
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    writer = NuRadioRecoio(filepath)
    
    for evt in events:
        writer.write_event(evt)
    
    writer.end()
    
    logger.info(f"Saved {len(events)} events to NUR file: {filepath}")


def save_correlation_map(corr_matrix, positions, evt, config, save_dir, **kwargs):
    """
    Save correlation map data to pickle file for later plotting.
    
    Parameters
    ----------
    corr_matrix : numpy.ndarray
        2D correlation matrix
    positions : dict
        Dictionary containing coordinate system info.
        Required keys: 'coord_system', 'rec_type', 'coord0_vec', 'coord1_vec'
    evt : NuRadio Event
        Event object
    config : dict
        Configuration dictionary
    save_dir : str
        Directory to save maps
    **kwargs : dict
        Additional data (alternate coordinates, exclusion zones, etc.)
    """
    station = evt.get_station()
    station_id = station.get_id()
    run_number = evt.get_run_number()
    event_number = evt.get_id()
    
    coord_system = positions['coord_system']
    rec_type = positions.get('rec_type', None)
    
    map_data = {
        'corr_matrix': corr_matrix,
        'station_id': station_id,
        'run_number': run_number,
        'event_number': event_number,
        'config': config,
        'coord_system': coord_system,
        'rec_type': rec_type,
        'limits': config['limits'],
        'step_sizes': config['step_sizes'],
        'fixed_coord': config['fixed_coord'],
        'channels': config['channels']
    }
    
    if 'coord0_alt' in kwargs and kwargs['coord0_alt'] is not None:
        map_data['coord0_alt'] = kwargs['coord0_alt']
        map_data['coord1_alt'] = kwargs['coord1_alt']
        map_data['alt_indices'] = kwargs.get('alt_indices')
    
    if 'exclusion_bounds' in kwargs and kwargs['exclusion_bounds'] is not None:
        map_data['exclusion_bounds'] = kwargs['exclusion_bounds']
    
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"station{station_id}_run{run_number}_evt{event_number}_corrmap.pkl"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(map_data, f)
    
    logger.debug(f"Saved correlation map to {filepath}")


def load_correlation_map(filepath):
    """
    Load correlation map data from pickle file.
    
    Parameters
    ----------
    filepath : str
        Path to pickle file
        
    Returns
    -------
    map_data : dict
        Dictionary containing correlation matrix and metadata
    """
    from NuRadioReco.utilities.io_utilities import read_pickle
    
    map_data = read_pickle(filepath)
    logger.debug(f"Loaded correlation map from {filepath}")
    
    return map_data
