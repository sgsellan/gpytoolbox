#include <igl/hausdorff.h>
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

void binding_hausdorff_distance(py::module& m) {
    m.def("_hausdorff_distance_cpp_impl",[](EigenDRef<MatrixXd> vt,
                         EigenDRef<MatrixXi> ft, EigenDRef<MatrixXd> ut,
                         EigenDRef<MatrixXi> gt)
        {
            double s;
            igl::hausdorff(vt,ft,ut,gt,s);
            return s;
        });
}
