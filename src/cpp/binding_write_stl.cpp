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

// npe_function(_write_stl_cpp_impl)
// npe_arg(file, std::string)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(binary, bool)
// npe_begin_code()
//     // npe_Matrix_v V(v), UV(uv), N(n);
//     // npe_Matrix_f F(f), Ft(ft), Fn(fn);
//     // return write_obj(file, V, F, UV, Ft, N, Fn);
//     return write_stl(file, v, f, binary);
// npe_end_code()
// //