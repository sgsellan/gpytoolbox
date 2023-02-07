// #include <npe.h>
// #include <pybind11/stl.h>
#include <igl/fast_winding_number.h>
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

void binding_fast_winding_number(py::module& m) {
    m.def("_fast_winding_number_cpp_impl",[](EigenDRef<MatrixXd> vt,
                         EigenDRef<MatrixXi> ft, EigenDRef<MatrixXd> qt)
        {
            Eigen::VectorXd S;
            igl::fast_winding_number(vt,ft,qt,S);
            return S;
        });
}

// // decimated_vertices,decimated_faces,J,I = igl.decimate(vertices,faces,num_faces)
// npe_function(_fast_winding_number_cpp_impl)
// npe_arg(vt, dense_double)
// npe_arg(ft, dense_int)
// npe_arg(qt, dense_double)
// npe_begin_code()
//     Eigen::VectorXd S;
//     // Eigen::MatrixXd V(vt);
//     // Eigen::MatrixXi F(ft);
//     // Eigen::MatrixXd Q(qt);
//     igl::fast_winding_number(vt,ft,qt,S);
//     return npe::move(S);
// npe_end_code()

