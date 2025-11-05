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

namespace py = pybind11;

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

// Main computation function
py::list compute_delay_matrices(
    py::array_t<int> channels,
    py::array_t<double> src_posn_enu_matrix,  // shape: (rows, cols, 3)
    py::dict ant_locs,  // dict[channel_id] -> [x, y, z]
    py::dict interpolators)  // dict[channel_id] -> scipy.interpolate.RegularGridInterpolator
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
    
    // Build C++ interpolators from Python scipy interpolators
    std::vector<FastBilinearInterpolator> cpp_interpolators;
    std::vector<std::vector<double>> cpp_ant_locs(n_channels);
    
    for (size_t i = 0; i < n_channels; ++i) {
        int ch = ch_ptr[i];
        
        // Get antenna location (use py::int_ for dict key)
        py::object loc_obj = ant_locs[py::int_(ch)];
        py::array_t<double> loc = loc_obj.cast<py::array_t<double>>();
        auto loc_buf = loc.request();
        double* loc_ptr = static_cast<double*>(loc_buf.ptr);
        cpp_ant_locs[i] = {loc_ptr[0], loc_ptr[1], loc_ptr[2]};
        
        // Get interpolator (use py::int_ for dict key)
        py::object interp = interpolators[py::int_(ch)];
        py::tuple grid = interp.attr("grid").cast<py::tuple>();
        py::array_t<double> grid_x = grid[0].cast<py::array_t<double>>();
        py::array_t<double> grid_y = grid[1].cast<py::array_t<double>>();
        py::array_t<double> values = interp.attr("values").cast<py::array_t<double>>();
        
        cpp_interpolators.emplace_back(grid_x, grid_y, values);
    }
    
    // Compute travel times for each channel
    std::vector<std::vector<double>> travel_times(n_channels, std::vector<double>(flat_size));
    
    for (size_t ch_idx = 0; ch_idx < n_channels; ++ch_idx) {
        const auto& pos = cpp_ant_locs[ch_idx];
        const auto& interp = cpp_interpolators[ch_idx];
        
        // Compute rho and interpolate
        for (size_t idx = 0; idx < flat_size; ++idx) {
            double dx = xy_flat[idx * 2] - pos[0];
            double dy = xy_flat[idx * 2 + 1] - pos[1];
            double rho = std::sqrt(dx * dx + dy * dy);
            
            travel_times[ch_idx][idx] = interp.interpolate(rho, z_flat[idx]);
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
    m.doc() = "Fast C++ implementation of delay matrix computation";
    
    m.def("compute_delay_matrices", &compute_delay_matrices,
          py::arg("channels"),
          py::arg("src_posn_enu_matrix"),
          py::arg("ant_locs"),
          py::arg("interpolators"),
          "Compute time delay matrices for all channel pairs using C++ for speed");
}
