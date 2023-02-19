#include "tinyply/tinyply_wrappers.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

void binding_read_ply(py::module& m) {
    m.def("_read_ply_cpp_impl",[](const std::string& filename)
        {
            Eigen::MatrixXd V, N;
            Eigen::MatrixXi F;
            Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> C;
            // read_ply_file("test/unit_tests_data/temp.ply");
            int s = read_ply(filename, V, F, N, C);
            // int s =0;
            return std::make_tuple(s, V, F, N, C);
        });
}