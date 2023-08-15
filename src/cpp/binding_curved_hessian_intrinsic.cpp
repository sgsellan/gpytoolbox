#include <igl/curved_hessian_energy.h>
#include <igl/orient_halfedges.h>
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

void binding_curved_hessian_intrinsic(py::module& m) {
    m.def("_curved_hessian_intrinsic_cpp_impl",[](EigenDRef<MatrixXd> l_sq,
                         EigenDRef<MatrixXi> f)
        {
            Eigen::MatrixXi E, oE;
            igl::orient_halfedges(f, E, oE);
            Eigen::SparseMatrix<double> Q;
            igl::curved_hessian_energy_intrinsic(f, l_sq, E, oE, Q);
            return Q;
        });
    
}
