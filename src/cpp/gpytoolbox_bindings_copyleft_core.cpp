#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

// using namespace Eigen;
namespace py = pybind11;
// using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
// template <typename MatrixType>
// using EigenDRef = Ref<MatrixType, 0, EigenDStride>; //allows passing column/row order matrices easily

//forward declare all bindings
void binding_swept_volume(py::module& m);
void binding_booleans(py::module& m);
void binding_tetrahedralize(py::module& m);


PYBIND11_MODULE(gpytoolbox_bindings_copyleft, m) {

    /// call all bindings declared above  
    binding_swept_volume(m);
    binding_booleans(m);
    binding_tetrahedralize(m);

    m.def("help", [&]() {printf("hi"); });
}

