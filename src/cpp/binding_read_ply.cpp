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
    m.def("_read_ply_cpp_impl",[](std::string filename)
        {
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            int s = read_ply(filename, V, F);
            return std::make_tuple(s, V, F);
        });
}