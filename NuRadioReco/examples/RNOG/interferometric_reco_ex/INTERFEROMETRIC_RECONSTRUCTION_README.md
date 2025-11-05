# Interferometric Direction Reconstruction

This module performs directional reconstruction of radio signals by fitting time delays between channels to pre-calculated time delay maps.

## Files in this Directory

- **`interferometric_reco_example.py`**: Simple reconstruction script with core functionality (recommended starting point)
- **`interferometric_reco_example_advanced.py`**: Advanced reconstruction script with additional features:
  - SNR-based channel filtering (`--snr-threshold`)
  - Edge signal detection (`--edge-sigma`)
  - Two-stage automatic reconstruction (`mode: 'auto'` in config)
  - Helper channel validation for quality control
- **`correlation_map_plotter.py`**: Standalone script for plotting saved correlation maps with comprehensive visualization options
- **`example_config.yaml`**: Example configuration file with all available options
- **`INTERFEROMETRIC_RECONSTRUCTION_README.md`**: This documentation file

### Which Script Should I Use?

**Use `interferometric_reco_example.py` (simple version) if:**
- You're new to the reconstruction module
- You want straightforward event processing
- Your data is clean and doesn't need quality filtering
- You're processing calibration pulser data or high-quality events

**Use `interferometric_reco_example_advanced.py` (advanced version) if:**
- You need automatic channel quality filtering based on SNR
- You want to detect and exclude channels with cut-off signals at trace edges
- You want fully automatic two-stage reconstruction (finds distance first, then direction)
- You're processing noisy data or need robust event-level quality cuts
- You want to skip events where no helper channels pass quality thresholds

Both scripts share the same configuration file format and output structures. The simple version is documented in the Quick Start section below, while the advanced features are detailed in the [Advanced Options](#advanced-options) section.

## Supporting Modules

The reconstruction uses several NuRadioReco utility modules:

- **`NuRadioReco.utilities.interferometry_io_utilities`**: I/O functions for saving/loading results and correlation maps
  - `save_interferometric_results_hdf5()`: Save reconstruction results to HDF5
  - `save_interferometric_results_nur()`: Save results to NUR format
  - `save_correlation_map()`: Save correlation map data to pickle
  - `load_correlation_map()`: Load correlation map from pickle
  - `create_organized_paths()`: Create organized directory structure for outputs

- **`NuRadioReco.utilities.caching_utilities`**: Caching system for delay matrices
  - Automatically caches computed delay matrices in `~/.cache/nuradio_delay_matrices/`
  - Cache key based on station, channels, grid parameters, and interpolation method
  - Significantly speeds up repeated runs with same configuration

## Overview

The interferometric direction reconstruction works by:
1. Computing cross-correlations between pairs of antenna channels
2. Comparing measured time delays to pre-calculated time delay maps
3. Finding the source location that maximizes the correlation

The core reconstruction functionality is implemented in the NuRadioReco module:
`NuRadioReco.modules.interferometricDirectionReconstruction`

## Table of Contents
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration File](#configuration-file)
- [Usage Examples](#usage-examples)
- [Output Formats](#output-formats)
- [Visualizing Results](#visualizing-results)
- [Coordinate Systems](#coordinate-systems)
- [Time Delay Tables](#time-delay-tables)
- [Advanced Options](#advanced-options)
  - [SNR-Based Channel Filtering](#snr-based-channel-filtering)
  - [Edge Signal Detection](#edge-signal-detection)
  - [Two-Stage Automatic Reconstruction](#two-stage-automatic-reconstruction)
  - [Alternate Reconstruction](#alternate-reconstruction)
  - [Plane Wave Fallback](#plane-wave-fallback)
  - [Signal Processing Options](#signal-processing-options)
  - [Caching](#caching)
- [Logging Configuration](#logging-configuration)
- [Troubleshooting](#troubleshooting)

---

## Requirements

### Pre-calculated Time Delay Tables
You **must** have pre-calculated time delay tables for your station and channels. These are `.npz` files containing travel time information for each channel.

**Location:** Tables should be in: `{time_delay_tables}/station{station_id}/st{station}_ch{channel}_rz_table.npz`

**Format:** Each table is a 2D grid of travel times as a function of (r, z) coordinates:
- **r**: perpendicular distance from the antenna (relative)
- **z**: absolute depth coordinate (NOT relative to antenna)

### Python Dependencies
- NuRadioReco
- Mattak
- numpy
- scipy
- matplotlib
- h5py
- pyyaml

**Note:** The detector configuration is automatically loaded from the RNO-G MongoDB database. No detector JSON file is needed.

### Optional: C++ Extension for Faster Performance

The reconstruction can optionally use a C++ extension to compute delay matrices ~2x faster. This is **highly recommended** for processing large datasets.

**Setup (one-time):**

```bash
# Navigate to the interferometric_reco_ex directory
cd /path/to/NuRadioReco/examples/RNOG/interferometric_reco_ex/

# Install pybind11 if not already installed
pip install pybind11

# Compile the C++ extension
python setup.py build_ext --inplace
```

That's it! The module will automatically detect and use the C++ extension when available.

**Verification:**

Run any reconstruction - you should see this message:
```
WARNING - C++ extension loaded successfully - will use fast C++ implementation for building time delay matrices
```

If you see "C++ extension not found - using Python implementation" instead, the compilation failed.

**Cluster Usage Note:**

If you're using a SLURM cluster with different CPU types across nodes, the compiled binary should still work on all nodes without recompilation. The `setup.py` is configured to use generic x86-64 instructions that are compatible with all modern processors.

---

## Quick Start

### 0. Try the Example First

To quickly see the reconstruction in action, before even looking at setting it up yourself in the next steps, first download the provided calibration pulser root file at /data/reconstruction/validation_sets/cal_pulsers/station21/run476/combined.root and the pre-generated time tables (at least for station 21 since this pulser run is from it) from /data/reconstruction/travel_times_analytic/ from the Chicago server. Once you've downloaded the root file and time tables, you can run the example script like so (make sure if you download the tables to any location other than a tables/ dir in this directory to change the default time_delay_tables setting in the example_config.yaml to the new location; same for specifying the input for combined.root below):

```bash
# Run reconstruction with example config and data
python interferometric_reco_example.py \
    --config example_config.yaml \
    --input combined.root \
    --events 7 \
    --save_maps \

# Plot the correlation map
python correlation_map_plotter.py \
    --input results/station21/run476/corr_map_data/station21_run476_evt7_corrmap.pkl \
    --minimaps
```

This should reproduce the example figure shown at `/data/reconstruction/example_plots/station21_run476_evt7_corrmap_phiz.png` on the Chicago server. The "combined.root" file used to reproduce this is from a calibration pulsing run with pulser on helper string C.

### 1. Create a Configuration File

Create a YAML file (e.g., `reco_config.yaml`) with your reconstruction parameters:

```yaml
# Coordinate system: "cylindrical" or "spherical"
coord_system: "cylindrical"

# Reconstruction type (depends on coord_system):
#   For cylindrical: "phiz" (azimuth + depth) or "rhoz" (radius + depth)
#   For spherical: not used (automatically uses azimuth + zenith)
rec_type: "phiz"

# Channels to use for reconstruction (list of antenna channel IDs)
channels: [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]

# Search grid limits: [coord0_min, coord0_max, coord1_min, coord1_max]
# Units depend on coordinate system:
#   phiz: [phi_min(deg), phi_max(deg), z_min(m), z_max(m)]
#   rhoz: [rho_min(m), rho_max(m), z_min(m), z_max(m)]
#   spherical: [phi_min(deg), phi_max(deg), theta_min(deg), theta_max(deg)]
limits: [0, 360, -200, 0]

# Step sizes for grid: [coord0_step, coord1_step]
# Same units as limits
step_sizes: [0.5, 0.5]

# Fixed coordinate (the coordinate not being reconstructed)
#   For phiz: fixed rho (radius) in meters
#   For rhoz: fixed phi (azimuth) in degrees
#   For spherical: fixed r (radial distance) in meters
fixed_coord: 125.0

# Station ID (required for processing and cache organization)
station_id: 21

# Path to time delay tables directory
time_delay_tables: "/path/to/time_delay_tables/"

# Interpolation method for time delay tables (optional - defaults to 'linear')
interp_method: "linear"          # Options: 'linear' or 'nearest'

# Output directory settings (optional - will use defaults if not specified)
save_results_to: "./results/"    # Base directory for all reconstruction data (default: "./results/")
                                 # Structure: {base}/station{ID}/run{NUM}/reco_data/ (results)
                                 #           {base}/station{ID}/run{NUM}/corr_map_data/ (correlation maps)

# Signal processing options
apply_cable_delay: true          # Apply cable delay correction (default: true)
                                 # WARNING: Only set to false if using preprocessed data 
                                 # that already has cable delays removed!
apply_upsampling: true           # Upsample waveforms to 5 GHz
apply_bandpass: false             # Apply bandpass filter
apply_cw_removal: false           # Remove CW interference
apply_hann_window: false          # Apply Hann window to correlations
use_hilbert_envelope: false       # Use Hilbert envelope for correlations

# Alternate reconstruction options
find_alternate_reco: false        # Find alternate reconstruction coordinates
alternate_exclude_radius_deg: 5.0 # Exclusion radius around primary maximum (degrees)

# CW removal parameters (if apply_cw_removal: true)
cw_freq_band: [0.1, 0.7]         # Frequency band in GHz
cw_peak_prominence: 4.0          # Peak prominence threshold
```

### 2. Run Reconstruction

**Basic usage:**
```bash
python interferometric_reco_example.py \
    --config reco_config.yaml \
    --input /path/to/data.root \
    --output_type hdf5
```

**Process specific events:**
```bash
python interferometric_reco_example.py \
    --config reco_config.yaml \
    --input /path/to/data.root \
    --events 7 10 15 \
    --save_maps \
```

**Multiple input files (same station):**
```bash
python interferometric_reco_example.py \
    --config reco_config.yaml \
    --input file1.root file2.root file3.root \
    --output_type hdf5
```

### 3. Visualize Correlation Maps (Optional)

If you saved correlation maps with `--save_maps`, use the separate plotting script:

```bash
# Plot a single correlation map
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/station21_run476_evt7_corrmap.pkl

# Plot all maps in a directory
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/

# Plot with minimaps enabled
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/ --minimaps

# Custom output directory
python correlation_map_plotter.py --input map.pkl --output custom_figures/

# Create comprehensive plot with waveforms and event information
python correlation_map_plotter.py \
    --input results/station21/run476/corr_map_data/station21_run476_evt7_corrmap.pkl \
    --comprehensive results/station21/run476/reco_data/station21_run476_reco_results.h5 \
    --minimaps
```

The `--comprehensive` option creates a multi-panel visualization including:
- Correlation map with reconstruction results
- Ray path visualization showing signal propagation
- Event information table with reconstruction parameters
- Waveform grid showing all channels used in reconstruction

---

## Configuration File

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `coord_system` | string | Coordinate system: "cylindrical" or "spherical" | `"cylindrical"` |
| `rec_type` | string | Type of reconstruction (only for cylindrical) | `"phiz"` |
| `channels` | list[int] | Antenna channels to use | `[0, 1, 2, 3]` |
| `limits` | list[float] | Search grid bounds [min0, max0, min1, max1] | `[0, 360, -200, 0]` |
| `step_sizes` | list[float] | Grid step sizes [step0, step1] | `[5, 5]` |
| `fixed_coord` | float | Value of fixed coordinate (not needed if `mode: 'auto'`) | `125.0` |
| `time_delay_tables` | string | Path to time delay tables directory | `"/path/to/tables/"` |
| `station_id` | int | Station ID for processing | `21` |

### Reconstruction Types

| `coord_system` | `rec_type` | What it reconstructs | Units |
|----------------|------------|---------------------|-------|
| `cylindrical` | `phiz` | Azimuth (φ) and depth (z), with fixed radius (ρ) | φ: degrees, z: meters |
| `cylindrical` | `rhoz` | Radius (ρ) and depth (z), with fixed azimuth (φ) | ρ: meters, z: meters |
| `spherical` | N/A | Azimuth (φ) and zenith (θ), with fixed radius (r) | φ, θ: degrees |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | `"manual"` | **Advanced script only.** Reconstruction mode: `"manual"` (standard single-stage) or `"auto"` (two-stage automatic). See [Two-Stage Automatic Reconstruction](#two-stage-automatic-reconstruction) |
| `plane_wave_fallback` | bool | `false` | **Advanced script only.** Enable plane wave fallback reconstruction when only one string has signal. See [Plane Wave Fallback](#plane-wave-fallback) |
| `apply_cable_delay` | bool | `true` | Apply cable delay correction. **WARNING:** Only disable if using preprocessed data with cable delays already removed! |
| `apply_upsampling` | bool | `false` | Upsample to 5 GHz |
| `apply_bandpass` | bool | `false` | Apply 100-600 MHz bandpass filter |
| `apply_cw_removal` | bool | `false` | Remove CW interference |
| `apply_hann_window` | bool | `false` | Apply Hann window to correlations |
| `use_hilbert_envelope` | bool | `false` | Use Hilbert envelope for correlations |
| `find_alternate_reco` | bool | `false` | Find alternate reconstruction coordinates |
| `alternate_exclude_radius_deg` | float | `5.0` | Exclusion radius around primary maximum (degrees) |
| `save_results_to` | string | `"./results/"` | Base directory for organized output structure |
| `cw_freq_band` | list[float] | `[0.1, 0.7]` | CW removal frequency band (GHz) |
| `cw_peak_prominence` | float | `4.0` | CW peak detection threshold |
| `interp_method` | string | `"linear"` | Interpolation method: 'linear' or 'nearest' |

---

## Usage Examples

### Example 1: Azimuth Reconstruction (phiz)

Reconstruct azimuth and depth with fixed perpendicular distance of 30m:

```yaml
# config_phiz.yaml
coord_system: "cylindrical"
rec_type: "phiz"
channels: [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
limits: [0, 360, -200, 0]        # φ: 0-360°, z: 0 to -200m
step_sizes: [0.5, 0.5]           # 0.5° in φ, 0.5m in z
fixed_coord: 30.0               # ρ = 30m
station_id: 23
time_delay_tables: "/path/to/tables/"
```

```bash
python interferometric_reco_example.py \
    --config config_phiz.yaml \
    --input station23_run1234.root \
    --output_type hdf5
```

### Example 2: Zenith Reconstruction (rhoz)

Reconstruct radius and depth with fixed azimuth:

```yaml
# config_rhoz.yaml
coord_system: "cylindrical"
rec_type: "rhoz"
channels: [0, 1, 2, 3, 5, 6, 7]
limits: [0, 200, -200, 0]        # ρ: 0-200m, z: 0 to -200m
step_sizes: [0.5, 0.5]           # 0.5m in both
fixed_coord: 0.0                 # φ = 0° (east) - doesn't matter if only using antennas on power string
station_id: 23
time_delay_tables: "/path/to/tables/"
```

### Example 3: Full Spherical

Reconstruct both azimuth and zenith with fixed distance:

```yaml
# config_spherical.yaml
coord_system: "spherical"
# rec_type not needed for spherical
channels: [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
limits: [0, 360, 0, 180]         # φ: 0-360°, θ: 0° (up) to 180° (down)
step_sizes: [0.5, 0.2]           # 0.5° in φ, 0.2° in θ
fixed_coord: 50.0               # r = 50m
station_id: 23
time_delay_tables: "/path/to/tables/"
```

### Example 4: With Signal Processing

Apply upsampling and CW removal:

```yaml
mode: "auto"
coord_system: "spherical"
rec_type: "phitheta"
channels: [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
limits: [0, 360, 0, 180]
step_sizes: [0.5, 0.5]
station_id: 23
time_delay_tables: "/path/to/tables/"

apply_upsampling: true           # Upsample to 5 GHz
apply_cw_removal: true           # Remove CW interference
apply_hann_window: false         # Apply Hann window to correlations
use_hilbert_envelope: true      # Use Hilbert envelope for correlations
find_alternate_reco: true        # Find alternate reconstruction coordinates
alternate_exclude_radius_deg: 20.0 # Exclusion radius around primary (degrees)
```

---

## Visualizing Results

### Basic Correlation Map Plots

The `correlation_map_plotter.py` script can create standalone correlation map visualizations:

```bash
# Plot a single correlation map
python correlation_map_plotter.py --input station21_run476_evt7_corrmap.pkl

# Plot all maps in a directory
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/

# Enable minimap insets (zoomed views around peaks)
python correlation_map_plotter.py --input map.pkl --minimaps

# Plot specific pattern
python correlation_map_plotter.py --input results/ --pattern "*run476*" --minimaps
```

### Comprehensive Event Visualization

For detailed event analysis, use the `--comprehensive` flag to create multi-panel plots combining:
- **Correlation map** with reconstruction results including alternate coordinates if enabled during the reconstruction
- **Ray path visualization** showing signal propagation from vertex to antennas
- **Event information table** with truth values from simulations
- **Waveform grid** displaying all channels used in reconstruction

```bash
python correlation_map_plotter.py \
    --input results/station21/run476/corr_map_data/station21_run476_evt7_corrmap.pkl \
    --comprehensive results/station21/run476/reco_data/station21_run476_reco_results.h5 \
    --minimaps
```

**Requirements for comprehensive plots:**
- Correlation map pickle file (saved with `--save_maps`)
- Reconstruction results HDF5 file (contains event metadata and data filename)
- Access to original data file (for waveform extraction)

### Channel Pair Correlation Grids

If you saved pairwise correlation maps with `--save-pair-maps`, you can visualize all channel pair correlations:

```bash
python correlation_map_plotter.py \
    --pair-grid results/station21/run476/corr_map_data/pairwise_maps/
```

This creates a grid showing individual correlation maps for each antenna pair.

### Visualizing Multi-Stage Reconstructions

When using `--save-maps both` with auto mode reconstruction, you can visualize both stages side-by-side:

```bash
# First, run reconstruction with multi-stage map saving
python interferometric_reco_example_advanced.py \
    --config auto_config.yaml \
    --input data.root \
    --save-maps both

# Then plot the multi-stage correlation maps (note: --multistage flag is required!)
python correlation_map_plotter.py \
    --input results/station23/run100/corr_map_data/station23_run100_evt5_corrmap_multistage.pkl \
    --multistage
```

**Important:** The `--multistage` flag is required when plotting multi-stage correlation maps. Without it, the plotter will not recognize the multi-stage file format.

This creates a two-panel plot showing:
- **Left panel:** Stage 1 (rhoz) correlation map showing the distance and depth finding
- **Right panel:** Stage 2 (spherical) correlation map showing the azimuth and zenith finding

The plot titles show which channels and fixed coordinates were used in each stage.

---

## Output Formats

The reconstruction automatically creates an organized directory structure:

```
results/
└── station{ID}/
    └── run{NUM}/
        ├── reco_data/
        │   └── station{ID}_run{NUM}_reco.h5  (or .nur)
        └── corr_map_data/  (if --save_maps used)
            ├── station{ID}_run{NUM}_evt{N}_corrmap.pkl
            └── ...
```

You can customize the base directory with the `save_results_to` config parameter.

### HDF5 Output (`.h5`)

Structured table with reconstruction results saved to `{base}/station{ID}/run{NUM}/reco_data/`:

```python
import h5py
import pandas as pd

# Read HDF5 file
with h5py.File('results/station21/run476/reco_data/station21_run476_reco.h5', 'r') as f:
    data = f['reconstruction'][:]
    config = dict(f.attrs)  # Configuration stored as attributes

df = pd.DataFrame(data)
print(df.columns)
# Columns depend on rec_type:
# - All: runNum, eventNum, maxCorr, surfCorr
# - phiz: phi, z, phi_alt, z_alt (if alternate reconstruction enabled)
# - rhoz: rho, z, rho_alt, z_alt (if alternate reconstruction enabled)
# - phitheta: phi, theta, phi_alt, theta_alt (if alternate reconstruction enabled)
```

**Column Descriptions:**
- `runNum`: Run number
- `eventNum`: Event ID
- `maxCorr`: Maximum correlation value (0-1, higher is better)
- `surfCorr`: Maximum correlation in top 10m (surface correlation)
- `phi`: Reconstructed azimuth in degrees (for phiz, phitheta)
- `phi_alt`: Alternate reconstructed azimuth in degrees (if find_alternate_reco enabled)
- `theta`: Reconstructed zenith in degrees (for spherical)
- `theta_alt`: Alternate reconstructed zenith in degrees (if find_alternate_reco enabled)
- `rho`: Reconstructed perpendicular distance in meters (for rhoz)
- `rho_alt`: Alternate reconstructed perpendicular distance in meters (if find_alternate_reco enabled)
- `z`: Reconstructed depth in meters (for phiz, rhoz)
- `z_alt`: Alternate reconstructed depth in meters (if find_alternate_reco enabled)

### NUR Output (`.nur`)

NuRadioReco event format with reconstruction parameters stored in station parameters:

```python
from NuRadioReco.modules.io.eventReader import eventReader

reader = eventReader()
reader.begin('output.nur', read_detector=True)

for event in reader.run():
    station = event.get_station()
    
    max_corr = station.get_parameter(stnp.rec_max_correlation)
    surf_corr = station.get_parameter(stnp.rec_surf_corr)
    coord0 = station.get_parameter(stnp.rec_coord0)
    coord1 = station.get_parameter(stnp.rec_coord1)
    
    # For phiz: coord0 = phi, coord1 = z
    # For rhoz: coord0 = rho, coord1 = z
    # For spherical: coord0 = phi, coord1 = theta
```

### Correlation Map Data

When using `--save_maps`, correlation map data is saved to pickle files in `{base}/station{ID}/run{NUM}/corr_map_data/`:

Each `.pkl` file contains:
- `corr_matrix`: 2D correlation map
- `station_id`, `run_number`, `event_number`: Event identifiers
- `config`: Reconstruction configuration
- `coord_system`, `rec_type`: Coordinate system information
- `limits`: Grid boundaries
- `channels`: Channels used
- `fixed_coord`: Fixed coordinate value
- `coord0_alt`, `coord1_alt`: Alternate reconstruction coordinates (if enabled)
- `exclusion_bounds`: Exclusion zone information (if alternate reco enabled)

These files can be visualized using `correlation_map_plotter.py` (see next section).

**Retrieving the configuration from saved maps:**

If you need to check what configuration was used to generate a correlation map (useful if you've forgotten your settings), you can load it back:

```python
from NuRadioReco.utilities.interferometry_io_utilities import load_correlation_map

# Load the saved correlation map
map_data = load_correlation_map('path/to/corrmap.pkl')

# Extract the full config dictionary
config = map_data['config']

# Access any config parameter
print(f"Coordinate system: {config['coord_system']}")
print(f"Reconstruction type: {config['rec_type']}")
print(f"Channels used: {config['channels']}")
print(f"Grid limits: {config['limits']}")
print(f"Step sizes: {config['step_sizes']}")
print(f"Fixed coordinate: {config['fixed_coord']}")
# ... and all other config parameters

# or just list them all like so:
print(config)
```

---

## Visualizing Results

### Plotting Correlation Maps

Use the standalone `correlation_map_plotter.py` script to visualize saved correlation maps:

```bash
# Plot a single event
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/station21_run476_evt7_corrmap.pkl

# Plot all events in a run
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/

# Enable minimap insets for detailed views
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/ --minimaps

# Save to custom directory
python correlation_map_plotter.py --input map.pkl --output my_figures/

# Process with pattern matching
python correlation_map_plotter.py --input results/station21/run476/corr_map_data/ --pattern "*evt7*"
```

**Output:** Plots are automatically saved to an organized structure:
- Default: `figures/station{ID}/run{NUM}/station{ID}_run{NUM}_evt{N}_corrmap.png`
- Custom: `{output_dir}/station{ID}/run{NUM}/station{ID}_run{NUM}_evt{N}_corrmap.png`

### Enhanced Plotting

The `correlation_map_plotter.py` script has options for enhanced visualizations with minimap insets.

**Note:** The `create_minimaps` parameter is used in the **plotter script**, not in the reconstruction config:

```bash
# Enable minimaps when plotting
python correlation_map_plotter.py --input corr_map_data/ --minimaps
```

It also allows for plotting an arbitrary number of additional points through the optional --extra-points argument. It expects 3 values: x value, y value, and label. This is most useful for adding a known location point to the map to compare to the reconstructed location. For example, let's say I know that my calibration pulser is at 182.1 degrees in azimuth and 87.5 m in depth with respect to the center of the phased array (the origin for the spherical case) and I want the known pulser location to be on the map. Then I would do:

```bash
# Enable extra point when plotting
python correlation_map_plotter.py --input corr_map_data/ --extra-points "182.1, 87.5, Pulser"
```

---

## Coordinate Systems

Coordinates are relative to the power string position at the surface.

### Cylindrical Coordinates

#### phiz Mode
- **Reconstructed:** φ (azimuth), z (depth)
- **Fixed:** ρ (perpendicular distance)
- **Use case:** When you know the approximate distance and want to find direction and depth

```
φ: Azimuth angle (0° = East, 90° = North, 180° = West, 270° = South)
z: Depth (negative values represent going under the surface, positive above)
ρ: Perpendicular distance from power string
```

#### rhoz Mode
- **Reconstructed:** ρ (perpendicular distance), z (depth)
- **Fixed:** φ (azimuth)
- **Use case:** When you know the direction and want to find distance and depth (zenith angle reconstruction)

```
ρ: Perpendicular distance from power string
z: Depth (negative values represent going under the surface, positive above)
φ: Fixed azimuth direction (0° = East, 90° = North, 180° = West, 270° = South)
```

### Spherical Coordinates

**For spherical coordinate system (no rec_type needed)**
- **Reconstructed:** φ (azimuth), θ (zenith angle)
- **Fixed:** r (radial distance)
- **Use case:** Full directional reconstruction with known distance

```
φ: Azimuth angle (0° = East, 90° = North, 180° = West, 270° = South)
θ: Zenith angle (0° = up, 90° = perpendicular, 180° = down)
r: Radial distance from power string at surface of the ice.
```

---

## Time Delay Tables

### Required Directory Structure

```
time_delay_tables/
└── station23/
    ├── ch0_rz_table_rel_ant.npz
    ├── ch1_rz_table_rel_ant.npz
    ├── ch2_rz_table_rel_ant.npz
    ├── ch3_rz_table_rel_ant.npz
    └── ...
```

### Table Format

Each `.npz` file must contain:
- `data`: 2D array of travel times (nanoseconds) with shape: (len(r_range_vals), len(z_range_vals))
- `r_range_vals`: 1D array of **perpendicular distances** from the antenna (meters)
- `z_range_vals`: 1D array of **absolute depths** (meters)

**Important coordinate details:**
- **r (perpendicular)**: Distance from antenna in the perpendicular plane (relative to antenna)
  - Example: r=100m means 100 meters away perpendicularly from the antenna
- **z (depth)**: **Absolute depth coordinate** (NOT relative to antenna)
  - Example: z=-50m means 50 meters below the ice surface (absolute depth)
  - z=0 is at the ice surface, z<0 is below surface (in the ice)

**Table interpretation:**
- `data[i, j]` = travel time from source at position `(ant_x + r[i], ant_y, z[j])` to the antenna
- The source is placed at perpendicular distance `r[i]` from the antenna, at absolute depth `z[j]`
- The horizontal position is offset by `r` from the antenna, but the vertical position is **absolute**, not relative to antenna depth

### Generating Time Delay Tables

Time delay tables can be generated using ray-tracing codes that account for:
- Ice refractivity profile (e.g., exponential density model for Greenland)
- Signal propagation paths (direct, refracted, and reflected rays)
- Antenna positions in station coordinate system
- The fastest arrival time is stored (minimum across all ray types)

**Example generation code:**
```python
# For each grid point (r, z):
src_position = antenna_position + [r, 0, 0]  # Offset perpendicularly by r
src_position[2] = z                          # Set to absolute depth z (not relative!)
travel_time = raytracer.get_travel_time(src_position, antenna_position)
```

## Advanced Options

### SNR-Based Channel Filtering

**Available in:** `interferometric_reco_example_advanced.py` only

The advanced script can automatically filter out low-SNR channels and skip events with insufficient signal quality.

**Usage:**
```bash
python interferometric_reco_example_advanced.py \
    --config config.yaml \
    --input data.root \
    --snr-threshold 2.0 \
```

**How it works:**
1. Calculates SNR for each channel
2. Drops channels below the specified threshold
3. Checks if at least one "helper channel" (channels 9, 10, 22, or 23) passes the threshold
4. Skips the entire event if no helper channels pass (indicates poor event quality)

**Helper channels:** Channels 9, 10, 22, 23 are on the so-called 'Helper' strings that are separate from the primary 'Power' string that the phased array antennas that are responsible for our trigger are on. At least one must have good SNR for reliable azimuthal reconstruction.

**Example output:**
```
Processing event 42:
  Channel SNRs: {0: 1.2, 1: 5.1, 2: 4.8, 3: 1.9, 9: 6.2, 10: 5.5}
    Channel 0 DROPPED: SNR too low (SNR=1.20 < threshold=2.00)
    Channel 3 DROPPED: SNR too low (SNR=1.90 < threshold=4.00)
  Summary: 4 channel(s) passed SNR threshold: [1, 2, 9, 10]
  Helper channels passing: [9, 10]
  Running reconstruction with 4 channels...
```

---
### Edge Signal Detection

**Available in:** `interferometric_reco_example_advanced.py` only

Detects and filters out channels where signals are cut off at the edges of the trace window. This can happen when very early/late arriving signals are partially outside the readout window.

**Usage:**
```bash
python interferometric_reco_example_advanced.py \
    --config config.yaml \
    --input data.root \
    --edge-sigma 3.0 \
```

**How it works:**
1. Divides each trace into chunks (default: 10 chunks)
2. Calculates RMS for each chunk
3. Compares edge chunk RMS to statistics from middle chunks
4. Flags channel as edge signal if edge RMS > median(middle) + N×std(middle)
5. Drops flagged channels from reconstruction
6. Similar to the SNR threshold, skips event if no helper channels remain


**Example output:**
```
Processing event 15:
  Edge detection for channel 2:
    First edge RMS: 12.3 mV, Last edge RMS: 45.8 mV
    Middle chunks: median=8.2 mV, std=2.1 mV
    Threshold: 14.5 mV
    EDGE DETECTED (last edge high)
  Summary: 1 channel(s) dropped due to edge signals: [2]
  Summary: 3 channel(s) passed edge detection: [0, 1, 3]
```

**Combining with SNR filtering:**
```bash
python interferometric_reco_example_advanced.py \
    --config config.yaml \
    --input data.root \
    --snr-threshold 2.0 \
    --edge-sigma 3.0 \
```

Both filters are applied sequentially: first edge detection, then SNR filtering. This ensures only clean, high-quality channels are used for reconstruction.

---

### Two-Stage Automatic Reconstruction

**Available in:** `interferometric_reco_example_advanced.py` only

The two-stage automatic mode performs a fully automatic reconstruction without needing to specify a fixed coordinate. It works in two stages:

1. **Stage 1 (rhoz mode):** Reconstruct perpendicular distance (ρ) and depth (z).
2. **Stage 2 (spherical mode):** Using the radial distance calculated from stage 1's ρ and z value, reconstruct azimuth (φ) and zenith (θ) to find the direction.

**Configuration:**
```yaml
# example_auto_config.yaml
mode: "auto"  # Enable two-stage automatic reconstruction

coord_system: "spherical"  # Final output will be in spherical coordinates
channels: [0, 1, 2, 3, 5, 6, 7, 9, 10]
step_sizes: [0.5, 0.5]  # Used in both stages
station_id: 23
time_delay_tables: "/path/to/tables/"

# Note: fixed_coord is NOT needed in auto mode
# limits will be set automatically for each stage
```

**Usage:**
```bash
python interferometric_reco_example_advanced.py \
    --config example_auto_config.yaml \
    --input data.root \
    --save-maps
```

**Saving Multi-Stage Correlation Maps:**

When using auto mode, you can save correlation maps from both stages in a single file for side-by-side visualization when plotting:

```bash
python interferometric_reco_example_advanced.py \
    --config example_auto_config.yaml \
    --input data.root \
    --save-maps both
```

Using `--save-maps both` creates files like `station{ID}_run{NUM}_evt{N}_corrmap_multistage.pkl` containing both stage 1 and stage 2 correlation maps. These can be visualized side-by-side using:

```bash
python correlation_map_plotter.py \
    --input results/station23/run100/corr_map_data/station23_run100_evt5_corrmap_multistage.pkl \
    --multistage
```

This creates a two-panel plot showing the coarse distance finding (stage 1 rhoz) and fine direction finding (stage 2 spherical) correlation maps together. See [Visualizing Multi-Stage Reconstructions](#visualizing-multi-stage-reconstructions) for more details.

**Note:** Using `--save-maps` (without `both`) in auto mode will only save the final stage 2 correlation map.

**How it works:**

**Stage 1 (rhoz reconstruction):**
- Searches over: ρ = [0, 200] m, z = [-200, 0] m using power string antennas [0, 1, 2, 3, 5, 6, 7]
- Fixed: φ = 0° (doesn't matter when using antennas on only 1 string like we are here)
- Finds: Best-fit perpendicular distance and depth
- Calculates: Radial distance r = √(ρ² + z²) from phased array center

**Stage 2 (spherical reconstruction):**
- Searches over: φ = [0, 360]°, θ = [0, 180]°
- Fixed: r = result from stage 1
- Finds: Best-fit azimuth and zenith angles
- This is the final reconstruction output

**Example output:**
```
[AUTO MODE] Stage 1: Running rhoz reconstruction to find optimal distance...
[AUTO MODE] Stage 1 results: rho=125.3m, z_abs=-87.2m, z_rel_PA=-89.5m, r=153.7m, maxCorr=0.847
[AUTO MODE] Stage 2: Running spherical reconstruction with fixed r=153.7m...
[AUTO MODE] Stage 2 results: phi=182.4°, theta=54.2°, maxCorr=0.863

=== Reconstruction Results ===
Station: 23
phi: 182.400°
theta: 54.200°
maxCorr: 0.863
surfCorr: 0.145
===============================
```

**When to use auto mode:**
- You don't know the approximate source distance
- You want fully automatic processing without parameter tuning
- You're scanning large datasets and want consistent methodology

**When to use manual mode:**
- You know the approximate source distance (e.g., calibration pulsers)
- You want to test specific geometric hypotheses
- You need to scan a specific region in detail
- You want to avoid the computational cost of two-stage reconstruction

**Performance notes:**
- Stage 1 uses a fixed grid (ρ: 0-200m, z: -200-0m) with power string channels [0, 1, 2, 3, 5, 6, 7]
- Correlation maps saved with `--save-maps` are from the final stage only (spherical)

---

### Alternate Reconstruction

**Available in:** Both simple and advanced scripts

The module can find alternate reconstruction coordinates by excluding regions around the primary maximum:

```yaml
find_alternate_reco: true        # Enable alternate coordinate finding
alternate_exclude_radius_deg: 5.0 # Exclusion radius around primary maximum
```

When enabled, alternate coordinates are saved in the HDF5 output as `phi_alt`, `z_alt`, etc., and displayed in plots. This is useful for:
- Identifying ambiguous reconstructions with multiple correlation peaks
- Understanding reconstruction uncertainties
- Detecting multi-path propagation scenarios

---

### Plane Wave Fallback

**Available in:** `interferometric_reco_example_advanced.py` only

When processing cosmic ray or neutrino events, you may encounter cases where only the antennas on one string (the power string with channels 0-3) detect a signal, while helper string antennas see only noise. In these cases, standard multi-string reconstruction fails because it requires signals on multiple strings for accurate azimuthal reconstruction.

The plane wave fallback mode provides a backup reconstruction strategy for these single-string events by:
1. Using only power string channels [0, 1, 2, 3]
2. Performing a 1D zenith angle scan (0° to 180°) with azimuth fixed at 0°
3. Assuming a plane wave approximation with fixed radius of 10m (encompassing all 4 antennas)

**Configuration:**
```yaml
plane_wave_fallback: true    # Enable plane wave fallback mode (default: false)
```

**Triggering conditions:**

Plane wave fallback is automatically triggered when:
- SNR filtering is enabled (`--snr-threshold`) AND no helper channels [9, 10, 22, 23] pass the threshold
- Edge filtering is enabled (`--edge-sigma`) AND no helper channels remain after filtering
- Both conditions apply if using both filters

**Example:**
```bash
python interferometric_reco_example_advanced.py \
    --config config_with_fallback.yaml \
    --input data.root \
    --snr-threshold 2.0 \
```

**Output characteristics:**
- **Azimuth (φ)**: Set to `NaN` to mark fallback reconstructions
- **Zenith (θ)**: Reconstructed value from 1D scan
- **Filtering in analysis**: Easy to identify fallback events with `df[np.isnan(df['phi'])]`

**Example output:**
```
Processing event 42:
  Channel SNRs: {0: 3.2, 1: 4.1, 2: 3.8, 3: 2.9, 9: 1.5, 10: 1.3, 22: 1.7, 23: 1.4}
    Channel 9 DROPPED: SNR too low (SNR=1.50 < threshold=2.00)
    Channel 10 DROPPED: SNR too low (SNR=1.30 < threshold=2.00)
    Channel 22 DROPPED: SNR too low (SNR=1.70 < threshold=2.00)
    Channel 23 DROPPED: SNR too low (SNR=1.40 < threshold=2.00)
  No helper channels passed SNR threshold - will attempt plane wave fallback
  [PLANE WAVE FALLBACK] No helper channels remaining - triggering fallback mode
[PLANE WAVE FALLBACK] Using channels [0,1,2,3] with fixed r=10m, azimuth=0°, scanning zenith 0-180°
[PLANE WAVE FALLBACK] Results: zenith=52.5°, maxCorr=0.783
```

**Analysis considerations:**
- Fallback reconstructions have `phi=NaN` for easy filtering
- Zenith angles are still meaningful and can provide useful directional information

---

### Signal Processing Options

**Available in:** Both simple and advanced scripts

Additional correlation analysis options:

```yaml
apply_hann_window: true          # Apply Hann window to reduce spectral leakage
use_hilbert_envelope: true       # Use envelope correlation for better SNR
```

**Hann window:** Reduces edge effects in correlation by tapering the trace edges.

**Hilbert envelope:** Uses signal envelope instead of raw waveform for correlation. Can improve reconstruction when:
- SNR is low
- Phase information is unreliable

---

### Caching

**Available in:** Both simple and advanced scripts

The module can use `NuRadioReco.utilities.caching_utilities` to cache delay matrices if you use the --use-cache argument like so:
```bash
python interferometric_reco_example.py \
    --config config.yaml \
    --input data.root \
    --use-cache 1
```

Cache details:
- **Cache location:** `~/.cache/nuradio_delay_matrices/`
- **Cache key:** Generated from station ID, channels, grid parameters, and interpolation method
- **Behavior:** Automatically loads from cache if available, significantly speeding up repeated runs
- **Cache management:** To force regeneration, delete the cache directory (by default at ~/.cache/nuradio) or specific cache files

The cache is usable when:
- Running the same configuration multiple times with a single fixed coordinate
- Testing different preprocessing options with the same reconstruction grid

### Simulation Truth Fixed Coordinate (NUR files only)

**Available in:** `interferometric_reco_example_advanced.py` only

When processing NUR simulation files, you can use the `--sim-truth-fixed-coord` flag to automatically set the fixed coordinate to the true simulation value for each event. This is useful for validation and debugging reconstruction performance:

```bash
python interferometric_reco_example_advanced.py \
    --config config.yaml \
    --input simulation.nur \
    --sim-truth-fixed-coord \
```

**How it works:**
- For **phiz** reconstruction (fixed ρ): Uses true perpendicular distance from simulation vertex to PA center
- For **rhoz** reconstruction (fixed φ): Uses true azimuth from PA center to simulation vertex
- For **spherical** reconstruction (fixed r): Uses true radial distance from PA center to simulation vertex

**Important notes:**
- Only works with NUR simulation files that contain shower information (`event.get_sim_showers()`)
- The fixed coordinate is calculated **per-event** from the true interaction vertex
- Delay matrix caching is disabled when using this flag (since fixed_coord varies per event)
- This mode is primarily for validation - it tells you "if I knew the correct fixed coordinate, how well could I reconstruct the other coordinates?"

**Example use case:**
To test if your phiz reconstruction can accurately find azimuth and depth when given the true radius:

```bash
python interferometric_reco_example_advanced.py \
    --config phiz_config.yaml \
    --input some_sim_file.nur \
    --sim-truth-fixed-coord \
    --save-maps \
```

This will reconstruct φ and z for each event using the true ρ value, allowing you to isolate reconstruction errors in the free parameters.

---

### Combining Advanced Features

**Available in:** `interferometric_reco_example_advanced.py` only

All advanced features can be combined for robust, fully automatic processing:

```bash
# Fully automatic processing with quality filters
python interferometric_reco_example_advanced.py \
    --config auto_config.yaml \
    --input data.root \
    --snr-threshold 2.0 \
    --edge-sigma 3.0 \
    --save-maps \
```

**Configuration file (auto_config.yaml):**
```yaml
mode: "auto"                     # Two-stage automatic reconstruction
coord_system: "spherical"        # Final output coordinate system
channels: [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]  # Include helper channels
step_sizes: [0.5, 0.5]
station_id: 23
time_delay_tables: "/path/to/tables/"

# Signal processing
apply_upsampling: true
apply_cw_removal: true
apply_hilbert_envelope: true
apply_bandpass: false
interp_method: "linear"

# Alternate reconstruction
find_alternate_reco: true
alternate_exclude_radius_deg: 20.0
```

**Processing flow:**
1. Load event and apply signal processing (upsampling, CW removal, etc.)
2. **Edge detection:** Drop channels with signals at trace edges
3. **SNR filtering:** Drop low-SNR channels, check helper channel requirement
4. If quality checks pass:
   - **Stage 1 (rhoz):** Find optimal perpendicular distance and depth
   - **Stage 2 (spherical):** Find direction using perpendicular distance and depth from stage 1
   - **Alternate reco:** Find second-best correlation peak in case of azimuthal degeneracy from only 2/3 strings seeing signals
5. Save results and correlation maps
```

### Parallel Processing

For processing many files, use job arrays:

```bash
#!/bin/bash
#SBATCH --array=0-99

FILES=(run_*.root)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

python interferometric_reco_example.py \
    --config config.yaml \
    --input $FILE \
    --output_type hdf5
```

---

## Logging Configuration

**Available in:** Both simple and advanced scripts

By default, the scripts are configured to show only WARNING level messages from most packages, but INFO level messages from the reconstruction modules. This reduces clutter from detector loading, MongoDB connections, and other imported packages while keeping reconstruction progress visible. To reduce clutter even further, you can restrict the INFO level from the reconstruction modules to WARNING instead as well.

**Current logging configuration** (in both scripts):
```python
from NuRadioReco.utilities.logging import set_general_log_level

# Set general logging to WARNING to suppress noisy packages
set_general_log_level(logging.WARNING)

# But set INFO level for the specific modules we want to see
logging.getLogger("NuRadioReco.modules.interferometricDirectionReconstruction").setLevel(logging.INFO)
logging.getLogger("NuRadioReco.utilities.interferometry_io_utilities").setLevel(logging.INFO)
```

**Logging Levels:**

| Level | Code | What you see |
|-------|------|--------------|
| **DEBUG** | `logging.DEBUG` or `10` | Everything: detailed algorithm steps, intermediate values, cache hits |
| **INFO** | `logging.INFO` or `20` | **Default for reconstruction:** Progress messages, stage results, fallback triggers |
| **WARNING** | `logging.WARNING` or `30` | **Default for other packages:** Only warnings and errors |
| **ERROR** | `logging.ERROR` or `40` | Only errors |

**To see MORE detail** (e.g., for debugging):

Edit the script to set DEBUG level for reconstruction:
```python
set_general_log_level(logging.WARNING)
logging.getLogger("NuRadioReco.modules.interferometricDirectionReconstruction").setLevel(logging.DEBUG)
```

You'll see additional messages like:
```
[DEBUG] Entering _correlator: 6 channel pairs, 6 delay matrices
[DEBUG] Correlation matrix shape: (400, 721), max_corr: 0.863
[DEBUG] Found delay matrices in memory cache for key abc123
```

**To see EVERYTHING** (very verbose, not recommended):

```python
set_general_log_level(logging.INFO)  # Show INFO from all packages
```

This will flood your output with detector queries, MongoDB connections, etc.:
```
[INFO] Query information for station 11 at 2022-10-01 00:00:00
[INFO] Query information for station 12 at 2022-10-01 00:00:00
[INFO] Query information for station 13 at 2022-10-01 00:00:00
... (many lines)
[INFO] Attempting to connect to the database ...
[INFO] ... connection to RNOG_hardware_v0 established
```

**To see LESS** (only errors):

```python
set_general_log_level(logging.ERROR)
logging.getLogger("NuRadioReco.modules.interferometricDirectionReconstruction").setLevel(logging.ERROR)
```

## Troubleshooting

**General debugging tip:** If you're having issues, try enabling DEBUG logging to see detailed information about what the reconstruction is doing. See the [Logging Configuration](#logging-configuration) section for details.

### "No results to save"
**Cause:** No events matched the selection criteria or all events failed processing.

**Solutions:**
- Check that event IDs exist in input file
- For `.nur` files, use run numbers (not event IDs)
- Remove `--events` flag to process all events
- If using `--snr-threshold` or `--edge-sigma`, try lowering thresholds or disabling filters
- Check if all events are being skipped due to failed helper channel criteria
- **Advanced script only:** Enable `plane_wave_fallback: true` to recover single-string events instead of skipping them

### No clear maximum correlation
**Cause:** Poor signal quality, wrong channels, or incorrect grid.

**Solutions:**
- **Enable DEBUG logging** to see correlation matrix details and intermediate values
- Check SNR of event (use quality cuts with `--snr-threshold` in advanced script)
- Verify channel selection
- Adjust search grid (`limits`, `step_sizes`)
- Try different `fixed_coord` value, or use `mode: 'auto'` (advanced script)
- Consider using `use_hilbert_envelope: true` for better correlation
- Check for edge signals with `--edge-sigma` (advanced script)

### Slow performance
**Cause:** Large search grids or missing cache.

**Solutions:**
- Use coarser grid (`step_sizes`)
- Enable caching with `--use-cache` if you're repeating the reconstruction with the same config many times
- Let cache build (first run is slower)
- **Check DEBUG logging** to see if delay matrices are being computed or loaded from cache
- Process fewer events at once
- For advanced script in auto mode: expect ~2× slower than manual mode

### "Invalid mode" or "mode must be 'manual' or 'auto'"
**Cause:** (Advanced script only) Config file has invalid `mode` parameter.

**Solutions:**
- Set `mode: "manual"` for standard single-stage reconstruction
- Set `mode: "auto"` for two-stage automatic reconstruction
- If omitted, defaults to `"manual"`
- Only advanced script supports auto mode

### Reconstruction fails with auto mode
**Cause:** (Advanced script only) Stage 1 or stage 2 reconstruction failed.

**Solutions:**
- **Enable DEBUG logging** to see which stage failed and why
- Check verbose output to see which stage failed
- Verify time delay tables exist for all channels
- Ensure channels list includes sufficient coverage (recommend: [0,1,2,3,5,6,7,9,10,22,23])
- Try manual mode first to verify basic functionality

### Too much logging output / Can't find my messages
**Cause:** INFO level enabled for all packages, flooding output with detector queries and database connections.

**Solutions:**
- Check logging configuration at top of script
- Default should be: `set_general_log_level(logging.WARNING)` with specific modules at INFO - set other modules to WARNING as well to restrict output
- See [Logging Configuration](#logging-configuration) section for details
- Don't set general level to INFO unless you need to debug package imports

### Events being skipped that should have signal
**Cause:** (Advanced script only) SNR or edge filtering too aggressive, no fallback enabled.

**Solutions:**
- Lower `--snr-threshold` or `--edge-sigma` values
- Enable `plane_wave_fallback: true` in config to recover single-string events
- Verify that at least one helper channel [9, 10, 22, 23] has good signal

---

## Command-Line Arguments Reference

### Simple Script (`interferometric_reco_example.py`)

```bash
python interferometric_reco_example.py [OPTIONS]

Required:
  --config CONFIG              Path to YAML configuration file
  --input FILE [FILE ...]  Input data file(s) (.root or .nur)

Optional:
  --output_type {hdf5,nur}     Output format (default: hdf5)
  --outputfile FILE            Manually specify output file path
  --events N [N ...]           Specific event IDs to process
  --runs N [N ...]             Specific run numbers (NUR files only)
  --use-cache                  Enable delay matrix caching
  --save-maps                  Save correlation map data
  --save-pair-maps             Save channel pair correlation maps
```

### Advanced Script (`interferometric_reco_example_advanced.py`)

```bash
python interferometric_reco_example_advanced.py [OPTIONS]

Required:
  --config CONFIG              Path to YAML configuration file
  --input FILE [FILE ...]  Input data file(s) (.root or .nur)

Optional:
  --output_type {hdf5,nur}     Output format (default: hdf5)
  --outputfile FILE            Manually specify output file path
  --events N [N ...]           Specific event IDs to process
  --runs N [N ...]             Specific run numbers (NUR files only)
  --use-cache                  Enable delay matrix caching
  --save-maps                  Save correlation map data
  --save-pair-maps             Save channel pair correlation maps

Advanced Options:
  --snr-threshold FLOAT        SNR threshold for channel filtering
                               Drops channels below threshold
                               Skips event if no helper channels [9,10,22,23] pass
  --edge-sigma FLOAT           Edge signal detection threshold (in std devs)
                               Drops channels with signals at trace edges
                               Skips event if no helper channels remain
  --sim-truth-fixed-coord      Use simulation truth for fixed coordinate
                               (NUR simulation files only, for validation)
```

---

## Contact & Support

For questions or issues:
- Review examples in this README
- Check NuRadioReco documentation
- Contact Bryan Hendricks (blh5615@psu.edu)

---

