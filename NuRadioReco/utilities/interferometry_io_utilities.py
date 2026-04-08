"""I/O utilities for interferometric reconstruction results."""

import os
import re
import pickle
import logging
import numpy as np
import argparse

logger = logging.getLogger('NuRadioReco.utilities.interferometry_io_utilities')

def parse_event_ids(s):
    """Parse comma-separated RUN:EVENT pairs from a CLI argument string.

    Args:
        s: String like "123:0,456:1".

    Returns:
        List of (run_number, event_id) integer tuples.
    """
    pairs = []
    for token in s.split(","):
        if ":" not in token:
            raise argparse.ArgumentTypeError(f"Bad token '{token}', expected RUN:EVENT")
        r, e = token.split(":")
        pairs.append((int(r), int(e)))
    return pairs

def create_organized_paths(config, event_id, output_type, ray_type_mode=None, use_run_in_path=True):
    """
    Create organized directory structure and file paths for results and maps.
    
    Directory structure:
        base/station{id}/coord_system/rec_type/run{id}/reco_data/
        base/station{id}/coord_system/rec_type/run{id}/corr_map_data/
    
    This groups runs by coordinate system for easy comparison, while keeping
    each run's reco results and correlation maps together.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    event_id : tuple
        Tuple containing (run_number, event_number)
    output_type : str
        Output file type ('hdf5' or 'nur')
    ray_type_mode : str, optional
        Ray type mode used for reconstruction (e.g., 'auto', 'direct', 'viscosity').
        If provided, adds mode subdirectory to paths.
    use_run_in_path : bool, optional
        If True, use 'run{number}' in path. If False, use 'multirun' (default: True)
    
    Returns
    -------
    tuple : (results_path, maps_dir)
        Full path to results file and directory for correlation maps
    """
    run_number, event_number = event_id
    
    results_base = config.get('save_results_to', './results/')
    station_id = config.get('station_id')
    coord_system = config.get('coord_system', 'cylindrical')
    rec_type = config.get('rec_type', 'phiz')
    
    # Determine run identifier based on whether we're processing a single run or multiple
    if use_run_in_path and run_number is not None:
        run_identifier = f"run{run_number}"
    else:
        run_identifier = "multirun"
    
    # Structure: base/station{id}/coord_system/rec_type/run{id}/
    # This groups all runs by coordinate system for easy comparison
    station_dir = os.path.join(results_base, f"station{station_id}")
    coord_subdir = os.path.join(station_dir, coord_system, rec_type)
    run_dir = os.path.join(coord_subdir, run_identifier)
    
    # Keep reco_data and corr_map_data together under each run
    reco_data_dir = os.path.join(run_dir, "reco_data")
    corr_map_dir = os.path.join(run_dir, "corr_map_data")
    
    # Add ray type mode subdirectory if provided
    if ray_type_mode is not None:
        reco_data_dir = os.path.join(reco_data_dir, ray_type_mode)
        corr_map_dir = os.path.join(corr_map_dir, ray_type_mode)
    
    # Only create reco_data_dir here; corr_map_dir is created by save_correlation_map when needed
    os.makedirs(reco_data_dir, exist_ok=True)
    
    extension = 'h5' if output_type == 'hdf5' else 'nur'
    
    # Build filename based on directory structure
    # If use_run_in_path=True (single event/run), include run and event in filename
    # If use_run_in_path=False (multirun), just use station ID
    if use_run_in_path and run_number is not None:
        # Single run/event case - include full details in filename
        if event_number is not None:
            results_filename = f"station{station_id}_run{run_number}_event{event_number}_reco_results.{extension}"
        else:
            results_filename = f"station{station_id}_run{run_number}_reco_results.{extension}"
    else:
        # Multirun case - generic filename
        results_filename = f"station{station_id}_reco_results.{extension}"
    
    results_path = os.path.join(reco_data_dir, results_filename)
    
    return results_path, corr_map_dir


def determine_plot_output_path(file_path, output_arg, station_id, event_id):
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
    event_id : tuple
        Tuple containing (run_number, event_number)
    
    Returns
    -------
    str
        Full path for output plot file
    """
    run_number, event_number = event_id
    
    plot_filename = f"station{station_id}_run{run_number}_event{event_number}_corrmap.png"
    dir_identifier = run_number
    
    if output_arg is None:
        output_dir = os.path.join("figures", f"station{station_id}", f"run{dir_identifier}")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, plot_filename)
    
    elif os.path.isdir(output_arg) or output_arg.endswith('/'):
        base_dir = output_arg.rstrip('/')
        output_dir = os.path.join(base_dir, "figures", f"station{station_id}", f"run{dir_identifier}")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, plot_filename)
    
    else:
        output_dir = os.path.dirname(output_arg)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return output_arg


def save_reco_results_hdf5(results, filepath, config):
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
            if value is None:
                config_group.attrs[key] = "null"
            elif isinstance(value, (list, tuple, np.ndarray)):
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
            
            # Handle string fields (like filename)
            if key == 'filename' or all(isinstance(v, str) for v in values if v is not None):
                # Create variable-length string dataset
                dt = h5py.string_dtype(encoding='utf-8')
                data = np.array([v if v is not None else "" for v in values], dtype=dt)
                results_group.create_dataset(key, data=data)
            
            elif key in ['station_id', 'run_number', 'event_number']:
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
    
    return filepath


def save_reco_results_nur(events, filepath, channels_only=None):
    """Save interferometric reconstruction results to NUR format.

    Args:
        events: List of NuRadio Event objects with reconstruction
            parameters stored.
        filepath: Path to output NUR file.
        channels_only: If provided, a list of channel IDs to keep.
            All other channels are removed before writing, reducing
            file size. Useful for saving only coherent waveform
            channels (e.g., [100, 101]) without the full raw traces.

    Returns:
        Output filepath.
    """
    from NuRadioReco.framework.event import Event
    from NuRadioReco.framework.station import Station
    from NuRadioReco.modules.io.eventWriter import eventWriter

    outdir = os.path.dirname(filepath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    writer = eventWriter()
    writer.begin(filepath)

    for event in events:
        if channels_only is not None:
            evt_out = Event(event.get_run_number(), event.get_id())
            for stn in event.get_stations():
                stn_out = Station(stn.get_id())
                for ch_id in channels_only:
                    if stn.has_channel(ch_id):
                        stn_out.add_channel(stn.get_channel(ch_id))
                evt_out.set_station(stn_out)
            writer.run(evt_out)
        else:
            writer.run(event)

    writer.end()

    logger.info(f"Saved {len(events)} events to NUR file: {filepath}")

    return filepath


def save_corr_map(corr_matrix, positions, event, config, save_dir, **kwargs):
    """
    Save correlation map data to pickle file for later plotting.
    
    Parameters
    ----------
    corr_matrix : numpy.ndarray
        2D correlation matrix
    positions : dict
        Dictionary containing coordinate system info.
        Required keys: 'coord_system', 'rec_type', 'coord_0_vec', 'coord_1_vec'
    event : NuRadio Event
        Event object
    config : dict
        Configuration dictionary
    save_dir : str
        Directory to save maps
    **kwargs : dict
        Additional data (alternate coordinates, exclusion zones, etc.)
    """
    station = event.get_station()
    station_id = station.get_id()
    run_number = event.get_run_number()
    event_number = event.get_id()
    
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

    # Save coordinate vectors (centers) if provided in positions dict
    if 'coord_0_vec' in positions and positions['coord_0_vec'] is not None:
        try:
            coord_0 = np.array(positions['coord_0_vec'])
            # convert to plain Python list for pickle stability
            map_data['coord_0_vec'] = coord_0.tolist()
        except Exception:
            map_data['coord_0_vec'] = positions['coord_0_vec']

    if 'coord_1_vec' in positions and positions['coord_1_vec'] is not None:
        try:
            coord_1 = np.array(positions['coord_1_vec'])
            map_data['coord_1_vec'] = coord_1.tolist()
        except Exception:
            map_data['coord_1_vec'] = positions['coord_1_vec']
    
    if 'coord_0_alt' in kwargs and kwargs['coord_0_alt'] is not None:
        # Store alternate coordinates in their original units (radians, meters)
        map_data['coord_0_alt'] = kwargs['coord_0_alt']
        map_data['coord_1_alt'] = kwargs['coord_1_alt']
        map_data['alt_indices'] = kwargs.get('alt_indices')
    
    if 'exclusion_bounds' in kwargs and kwargs['exclusion_bounds'] is not None:
        map_data['exclusion_bounds'] = kwargs['exclusion_bounds']
    
    # Store channel pair info if this is a pairwise correlation map
    if 'pair_channels' in kwargs and kwargs['pair_channels'] is not None:
        map_data['pair_channels'] = kwargs['pair_channels']

    # If reconstruction coordinates and max are provided, save in original units (radians, meters)
    rec0 = kwargs.get('rec_coord_0', None)
    rec1 = kwargs.get('rec_coord_1', None)
    rec_max = kwargs.get('rec_max_corr', None)
    if rec0 is not None and rec1 is not None:
        map_data['coord_0'] = rec0
        map_data['coord_1'] = rec1
        map_data['max_corr'] = float(rec_max) if rec_max is not None else None
    
    # If pairwise reconstruction coordinates are provided, save them too
    pair_rec0 = kwargs.get('pair_rec_coord_0', None)
    pair_rec1 = kwargs.get('pair_rec_coord_1', None)
    pair_rec_max = kwargs.get('pair_rec_max_corr', None)
    if pair_rec0 is not None and pair_rec1 is not None:
        map_data['pair_rec_coord_0'] = pair_rec0
        map_data['pair_rec_coord_1'] = pair_rec1
        map_data['pair_rec_max_corr'] = float(pair_rec_max) if pair_rec_max is not None else None
    
    # Store ray_type_mode if provided (for reference, not for path organization)
    ray_type_mode = kwargs.get('ray_type_mode', None)
    if ray_type_mode is not None:
        map_data['ray_type_mode'] = ray_type_mode
    
    # Store time delays at best reconstruction position (for waveform alignment visualization)
    best_time_delays = kwargs.get('best_time_delays', None)
    if best_time_delays is not None:
        map_data['best_time_delays'] = best_time_delays
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Build filename with optional suffix (e.g., for channel pair info)
    filename_base = f"station{station_id}_run{run_number}_event{event_number}"
    filename_suffix = kwargs.get('filename_suffix', '')
    if filename_suffix:
        filename = f"{filename_base}{filename_suffix}_corrmap.pkl"
    else:
        filename = f"{filename_base}_corrmap.pkl"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(map_data, f)
    
    logger.debug(f"Saved correlation map to {filepath}")

    return filepath

def load_corr_map(filepath):
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
