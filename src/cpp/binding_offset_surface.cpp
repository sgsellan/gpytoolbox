
#include <igl/offset_surface.h>
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

void binding_offset_surface(py::module& m) {
    m.def("_offset_surface_cpp_impl",[](EigenDRef<MatrixXd> vt, EigenDRef<MatrixXi> ft, double iso, int grid_size)
        {
            Eigen::MatrixXd V(vt);
            Eigen::MatrixXi F(ft);
            Eigen::MatrixXd VB;
            Eigen::MatrixXi FB;
            Eigen::MatrixXd GV;
            Eigen::RowVector3i side;
            Eigen::VectorXd S;
            igl::offset_surface(V,F,iso,grid_size,igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,VB,FB,GV,side,S);
            return std::make_tuple(VB,FB);
        });
}