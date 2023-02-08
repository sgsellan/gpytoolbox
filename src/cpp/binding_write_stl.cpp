#include "microstl/microstl_wrappers.h"
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

void binding_write_stl(py::module& m) {
    m.def("_write_stl_cpp_impl",[](std::string filename, EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f,bool binary)
        {
            return write_stl(filename, v, f, binary);
        });
}