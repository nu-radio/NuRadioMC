/*
 * Python extension module for fast delay matrix computation using C++
 * 
 * Build with: python setup.py build_ext --inplace
 * Use in Python: import fast_delay_matrices
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

namespace py = pybind11;

// Fast 2D nearest-neighbor interpolator
class FastNearestInterpolator {
private:
    double x_min, y_min;
    double dx_inv, dy_inv;
    size_t nx, ny;
    std::vector<double> values_flat;
    
public:
    FastNearestInterpolator(
        py::array_t<double> grid_x,
        py::array_t<double> grid_y,
        py::array_t<double> values)
    {
        auto x_buf = grid_x.request();
        auto y_buf = grid_y.request();
        auto v_buf = values.request();
        
        nx = x_buf.shape[0];
        ny = y_buf.shape[0];
        
        double* x_ptr = static_cast<double*>(x_buf.ptr);
        double* y_ptr = static_cast<double*>(y_buf.ptr);
        double* v_ptr = static_cast<double*>(v_buf.ptr);
        
        x_min = x_ptr[0];
        y_min = y_ptr[0];
        dx_inv = 1.0 / (x_ptr[1] - x_ptr[0]);
        dy_inv = 1.0 / (y_ptr[1] - y_ptr[0]);
        
        // Copy values
        values_flat.assign(v_ptr, v_ptr + (nx * ny));
    }
    
    double interpolate(double x, double y) const {
        // Round to nearest grid point
        double xi = (x - x_min) * dx_inv;
        double yi = (y - y_min) * dy_inv;
        
        int i = static_cast<int>(std::round(xi));
        int j = static_cast<int>(std::round(yi));
        
        // Clamp to valid range
        i = std::max(0, std::min(i, static_cast<int>(nx) - 1));
        j = std::max(0, std::min(j, static_cast<int>(ny) - 1));
        
        return values_flat[i * ny + j];
    }
};

// Fast 2D bilinear interpolator
class FastBilinearInterpolator {
private:
    double x_min, y_min;
    double dx_inv, dy_inv;
    size_t nx, ny;
    std::vector<double> values_flat;
    
public:
    FastBilinearInterpolator(
        py::array_t<double> grid_x,
        py::array_t<double> grid_y,
        py::array_t<double> values)
    {
        auto x_buf = grid_x.request();
        auto y_buf = grid_y.request();
        auto v_buf = values.request();
        
        nx = x_buf.shape[0];
        ny = y_buf.shape[0];
        
        double* x_ptr = static_cast<double*>(x_buf.ptr);
        double* y_ptr = static_cast<double*>(y_buf.ptr);
        double* v_ptr = static_cast<double*>(v_buf.ptr);
        
        x_min = x_ptr[0];
        y_min = y_ptr[0];
        dx_inv = 1.0 / (x_ptr[1] - x_ptr[0]);
        dy_inv = 1.0 / (y_ptr[1] - y_ptr[0]);
        
        // Copy values
        values_flat.assign(v_ptr, v_ptr + (nx * ny));
    }
    
    double interpolate(double x, double y) const {
        double xi = (x - x_min) * dx_inv;
        double yi = (y - y_min) * dy_inv;
        
        int i0 = static_cast<int>(xi);
        int j0 = static_cast<int>(yi);
        
        i0 = std::max(0, std::min(i0, static_cast<int>(nx) - 2));
        j0 = std::max(0, std::min(j0, static_cast<int>(ny) - 2));
        
        int i1 = i0 + 1;
        int j1 = j0 + 1;
        
        double fx = xi - i0;
        double fy = yi - j0;
        
        double w00 = (1 - fx) * (1 - fy);
        double w10 = fx * (1 - fy);
        double w01 = (1 - fx) * fy;
        double w11 = fx * fy;
        
        return w00 * values_flat[i0 * ny + j0] + 
               w10 * values_flat[i1 * ny + j0] + 
               w01 * values_flat[i0 * ny + j1] + 
               w11 * values_flat[i1 * ny + j1];
    }
};

// Fast 2D bicubic interpolator
class FastBicubicInterpolator {
private:
    double x_min, y_min;
    double dx_inv, dy_inv;
    size_t nx, ny;
    std::vector<double> values_flat;
    
    // Cubic interpolation along one dimension
    double cubic_interp(double t, double v0, double v1, double v2, double v3) const {
        // Catmull-Rom spline (commonly used for bicubic)
        double t2 = t * t;
        double t3 = t2 * t;
        
        return 0.5 * ((2.0 * v1) +
                     (-v0 + v2) * t +
                     (2.0*v0 - 5.0*v1 + 4.0*v2 - v3) * t2 +
                     (-v0 + 3.0*v1 - 3.0*v2 + v3) * t3);
    }
    
    // Get value with boundary handling (clamp to edge)
    double get_value(int i, int j) const {
        i = std::max(0, std::min(i, static_cast<int>(nx) - 1));
        j = std::max(0, std::min(j, static_cast<int>(ny) - 1));
        return values_flat[i * ny + j];
    }
    
public:
    FastBicubicInterpolator(
        py::array_t<double> grid_x,
        py::array_t<double> grid_y,
        py::array_t<double> values)
    {
        auto x_buf = grid_x.request();
        auto y_buf = grid_y.request();
        auto v_buf = values.request();
        
        nx = x_buf.shape[0];
        ny = y_buf.shape[0];
        
        double* x_ptr = static_cast<double*>(x_buf.ptr);
        double* y_ptr = static_cast<double*>(y_buf.ptr);
        double* v_ptr = static_cast<double*>(v_buf.ptr);
        
        x_min = x_ptr[0];
        y_min = y_ptr[0];
        dx_inv = 1.0 / (x_ptr[1] - x_ptr[0]);
        dy_inv = 1.0 / (y_ptr[1] - y_ptr[0]);
        
        // Copy values
        values_flat.assign(v_ptr, v_ptr + (nx * ny));
    }
    
    double interpolate(double x, double y) const {
        double xi = (x - x_min) * dx_inv;
        double yi = (y - y_min) * dy_inv;
        
        int i1 = static_cast<int>(std::floor(xi));
        int j1 = static_cast<int>(std::floor(yi));
        
        // Clamp to valid range for the center points
        i1 = std::max(0, std::min(i1, static_cast<int>(nx) - 2));
        j1 = std::max(0, std::min(j1, static_cast<int>(ny) - 2));
        
        double fx = xi - i1;
        double fy = yi - j1;
        
        // Get 4x4 grid of surrounding points
        int i0 = i1 - 1;
        int i2 = i1 + 1;
        int i3 = i1 + 2;
        
        int j0 = j1 - 1;
        int j2 = j1 + 1;
        int j3 = j1 + 2;
        
        // Interpolate in x direction for each of 4 rows
        double v0 = cubic_interp(fx, get_value(i0, j0), get_value(i1, j0), get_value(i2, j0), get_value(i3, j0));
        double v1 = cubic_interp(fx, get_value(i0, j1), get_value(i1, j1), get_value(i2, j1), get_value(i3, j1));
        double v2 = cubic_interp(fx, get_value(i0, j2), get_value(i1, j2), get_value(i2, j2), get_value(i3, j2));
        double v3 = cubic_interp(fx, get_value(i0, j3), get_value(i1, j3), get_value(i2, j3), get_value(i3, j3));
        
        // Interpolate in y direction
        return cubic_interp(fy, v0, v1, v2, v3);
    }
};

// Main computation function
py::list compute_delay_matrices(
    py::array_t<int> channels,
    py::array_t<double> src_posn_enu_matrix,  // shape: (rows, cols, 3)
    py::dict ant_locs,  // dict[channel_id] -> [x, y, z]
    py::dict interpolators,  // dict[channel_id] -> scipy.interpolate.RegularGridInterpolator or raw dict
    std::string interp_method = "linear")  // 'nearest', 'linear', or 'cubic'
{
    auto ch_buf = channels.request();
    auto pos_buf = src_posn_enu_matrix.request();
    
    if (pos_buf.ndim != 3 || pos_buf.shape[2] != 3) {
        throw std::runtime_error("src_posn_enu_matrix must have shape (rows, cols, 3)");
    }
    
    size_t grid_rows = pos_buf.shape[0];
    size_t grid_cols = pos_buf.shape[1];
    size_t flat_size = grid_rows * grid_cols;
    
    int* ch_ptr = static_cast<int*>(ch_buf.ptr);
    size_t n_channels = ch_buf.shape[0];
    
    double* pos_ptr = static_cast<double*>(pos_buf.ptr);
    
    // Extract XY positions and Z grid
    std::vector<double> xy_flat(flat_size * 2);
    std::vector<double> z_flat(flat_size);
    
    for (size_t i = 0; i < grid_rows; ++i) {
        for (size_t j = 0; j < grid_cols; ++j) {
            size_t idx = (i * grid_cols + j);
            xy_flat[idx * 2] = pos_ptr[(i * grid_cols + j) * 3 + 0];      // x
            xy_flat[idx * 2 + 1] = pos_ptr[(i * grid_cols + j) * 3 + 1];  // y
            z_flat[idx] = pos_ptr[(i * grid_cols + j) * 3 + 2];            // z
        }
    }
    
    // Build C++ interpolators from Python data
    // Expects interpolators dict to contain either:
    //   - Raw dict with 'r_grid', 'z_grid', 'values' keys (optimized path)
    //   - scipy RegularGridInterpolator objects (backward compatibility)
    
    // Use function wrappers to handle different interpolator types
    std::vector<std::function<double(double, double)>> interp_funcs;
    std::vector<std::vector<double>> cpp_ant_locs(n_channels);
    
    // Storage for interpolators (must persist for the duration of computation)
    std::vector<FastNearestInterpolator> nearest_interps;
    std::vector<FastBilinearInterpolator> bilinear_interps;
    std::vector<FastBicubicInterpolator> bicubic_interps;
    
    if (interp_method == "nearest") {
        nearest_interps.reserve(n_channels);
    } else if (interp_method == "linear") {
        bilinear_interps.reserve(n_channels);
    } else if (interp_method == "cubic") {
        bicubic_interps.reserve(n_channels);
    } else {
        throw std::runtime_error("Invalid interp_method. Must be 'nearest', 'linear', or 'cubic'");
    }
    
    for (size_t i = 0; i < n_channels; ++i) {
        int ch = ch_ptr[i];
        
        // Get antenna location (use py::int_ for dict key)
        py::object loc_obj = ant_locs[py::int_(ch)];
        py::array_t<double> loc = loc_obj.cast<py::array_t<double>>();
        auto loc_buf = loc.request();
        double* loc_ptr = static_cast<double*>(loc_buf.ptr);
        cpp_ant_locs[i] = {loc_ptr[0], loc_ptr[1], loc_ptr[2]};
        
        // Get interpolator data (use py::int_ for dict key)
        py::object interp_obj = interpolators[py::int_(ch)];
        
        py::array_t<double> grid_x, grid_y, values;
        
        // Check if it's a dict (optimized raw array path) or scipy object (legacy)
        if (py::isinstance<py::dict>(interp_obj)) {
            // Fast path: Extract raw arrays from dict
            py::dict interp_dict = interp_obj.cast<py::dict>();
            grid_x = interp_dict["r_grid"].cast<py::array_t<double>>();
            grid_y = interp_dict["z_grid"].cast<py::array_t<double>>();
            values = interp_dict["values"].cast<py::array_t<double>>();
        } else {
            // Legacy path: Extract from scipy RegularGridInterpolator
            py::tuple grid = interp_obj.attr("grid").cast<py::tuple>();
            grid_x = grid[0].cast<py::array_t<double>>();
            grid_y = grid[1].cast<py::array_t<double>>();
            values = interp_obj.attr("values").cast<py::array_t<double>>();
        }
        
        // Build interpolator of the appropriate type
        if (interp_method == "nearest") {
            nearest_interps.emplace_back(grid_x, grid_y, values);
            const auto& interp_ref = nearest_interps.back();
            interp_funcs.push_back([&interp_ref](double x, double y) {
                return interp_ref.interpolate(x, y);
            });
        } else if (interp_method == "linear") {
            bilinear_interps.emplace_back(grid_x, grid_y, values);
            const auto& interp_ref = bilinear_interps.back();
            interp_funcs.push_back([&interp_ref](double x, double y) {
                return interp_ref.interpolate(x, y);
            });
        } else {  // cubic
            bicubic_interps.emplace_back(grid_x, grid_y, values);
            const auto& interp_ref = bicubic_interps.back();
            interp_funcs.push_back([&interp_ref](double x, double y) {
                return interp_ref.interpolate(x, y);
            });
        }
    }
    
    // Compute travel times for each channel
    std::vector<std::vector<double>> travel_times(n_channels, std::vector<double>(flat_size));
    
    for (size_t ch_idx = 0; ch_idx < n_channels; ++ch_idx) {
        const auto& pos = cpp_ant_locs[ch_idx];
        const auto& interp = interp_funcs[ch_idx];
        
        // Compute rho and interpolate
        for (size_t idx = 0; idx < flat_size; ++idx) {
            double dx = xy_flat[idx * 2] - pos[0];
            double dy = xy_flat[idx * 2 + 1] - pos[1];
            double rho = std::sqrt(dx * dx + dy * dy);
            
            travel_times[ch_idx][idx] = interp(rho, z_flat[idx]);
        }
    }
    
    // Compute pairwise differences and return as Python list of numpy arrays
    py::list result;
    for (size_t i = 0; i < n_channels; ++i) {
        for (size_t j = i + 1; j < n_channels; ++j) {
            py::array_t<double> delay_matrix({grid_rows, grid_cols});
            auto buf = delay_matrix.request();
            double* out_ptr = static_cast<double*>(buf.ptr);
            
            for (size_t idx = 0; idx < flat_size; ++idx) {
                out_ptr[idx] = travel_times[i][idx] - travel_times[j][idx];
            }
            
            result.append(delay_matrix);
        }
    }
    
    return result;
}

PYBIND11_MODULE(fast_delay_matrices, m) {
    m.doc() = "Fast C++ implementation of delay matrix computation with multiple interpolation methods";
    
    m.def("compute_delay_matrices", &compute_delay_matrices,
          py::arg("channels"),
          py::arg("src_posn_enu_matrix"),
          py::arg("ant_locs"),
          py::arg("interpolators"),
          py::arg("interp_method") = "linear",
          "Compute time delay matrices for all channel pairs using C++ for speed.\n\n"
          "Parameters:\n"
          "  channels: Array of channel IDs\n"
          "  src_posn_enu_matrix: Source position matrix (rows, cols, 3)\n"
          "  ant_locs: Dict mapping channel_id -> [x, y, z]\n"
          "  interpolators: Dict mapping channel_id -> interpolator or raw dict\n"
          "  interp_method: Interpolation method ('nearest', 'linear', or 'cubic')\n");
}
