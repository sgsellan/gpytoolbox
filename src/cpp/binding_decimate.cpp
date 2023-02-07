#include <igl/decimate.h>
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

void binding_decimate(py::module& m) {
    m.def("_decimate_cpp_impl",[](EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f, int num_faces)
        {
            Eigen::MatrixXd SV;
            Eigen::MatrixXi SF;
            Eigen::VectorXi J, I;
            igl::decimate(v,f,num_faces,SV,SF,I,J);
            return std::make_tuple(SV,SF,I,J);
        });
    
}
