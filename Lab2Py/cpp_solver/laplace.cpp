#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

py::array_t<double> solve_cpp(int n, double eps, int max_iter, 
                              py::array_t<double> init_grid) {
    py::buffer_info buf = init_grid.request();
    double* ptr = static_cast<double*>(buf.ptr);
    int rows = buf.shape[0];
    int cols = buf.shape[1];


    for (int k = 0; k < max_iter; ++k) {
        double max_diff = 0.0;
        
        for (int i = 1; i < rows - 1; ++i) {
            for (int j = 1; j < cols - 1; ++j) {

                double old_val = ptr[i * cols + j];
                double new_val = 0.25 * (ptr[(i + 1) * cols + j] + 
                                         ptr[(i - 1) * cols + j] + 
                                         ptr[i * cols + (j + 1)] + 
                                         ptr[i * cols + (j - 1)]);
                
                ptr[i * cols + j] = new_val;
                
                double diff = std::abs(new_val - old_val);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }
        
        if (max_diff < eps) {
            break;
        }
    }

    return init_grid;
}

PYBIND11_MODULE(cpp_laplace, m) {
    m.doc() = "Laplace solver using Gauss-Seidel method implemented in C++";
    m.def("solve", &solve_cpp, "Solve Laplace equation",
          py::arg("n"), py::arg("eps"), py::arg("max_iter"), py::arg("init_grid"));
}