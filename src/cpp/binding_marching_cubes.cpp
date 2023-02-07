#include <igl/marching_cubes.h>
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

void binding_marching_cubes(py::module& m) {
    m.def("_marching_cubes_cpp_impl",[](EigenDRef<VectorXd> S, EigenDRef<MatrixXd> GV, int nx, int nz, int ny, double iso)
        {
            Eigen::MatrixXi F;
            Eigen::MatrixXd V;
            igl::marching_cubes(S, GV, nx, ny, nz, iso, V, F);
            return std::tuple(V,F);
        });
}