#include <igl/fast_winding_number.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <memory>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;

// Wrapper that owns an igl::FastWindingNumberBVH precomputation. The BVH is
// built once at construction and reused on every winding_number query.
class FastWindingNumberBVHWrapper {
public:
    FastWindingNumberBVHWrapper(const Eigen::MatrixXd& V,
                                const Eigen::MatrixXi& F,
                                int order)
        : order_(order) {
        igl::fast_winding_number(V, F, order_, bvh_);
    }

    Eigen::VectorXd winding_number(const Eigen::MatrixXd& Q,
                                   float accuracy_scale) const {
        Eigen::VectorXd W;
        igl::fast_winding_number(bvh_, accuracy_scale, Q, W);
        return W;
    }

    int order() const { return order_; }

private:
    igl::FastWindingNumberBVH bvh_;
    int order_;
};

void binding_fast_winding_number_bvh(py::module& m) {
    py::class_<FastWindingNumberBVHWrapper>(m, "_FastWindingNumberBVH_cpp_impl")
        .def(py::init([](EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> F, int order) {
                 return new FastWindingNumberBVHWrapper(
                     Eigen::MatrixXd(V), Eigen::MatrixXi(F), order);
             }),
             py::arg("V"), py::arg("F"), py::arg("order") = 2)
        .def("winding_number",
             [](const FastWindingNumberBVHWrapper& self,
                EigenDRef<MatrixXd> Q,
                float accuracy_scale) {
                 return self.winding_number(Eigen::MatrixXd(Q), accuracy_scale);
             },
             py::arg("Q"), py::arg("accuracy_scale") = 2.0f)
        .def_property_readonly("order", &FastWindingNumberBVHWrapper::order);
}
