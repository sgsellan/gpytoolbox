#include <igl/AABB.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <memory>
#include <stdexcept>
#include <string>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;

// Wrapper that owns V, F and the libigl AABB tree (in either 2D or 3D form).
// Owning V and F is important: igl::AABB stores references to them and reuses
// them on every squared_distance query.
class AABBTreeWrapper {
public:
    AABBTreeWrapper(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
        : V_(V), F_(F), dim_(static_cast<int>(V.cols())) {
        if (dim_ == 2) {
            tree2_ = std::make_unique<igl::AABB<Eigen::MatrixXd, 2>>();
            tree2_->init(V_, F_);
        } else if (dim_ == 3) {
            tree3_ = std::make_unique<igl::AABB<Eigen::MatrixXd, 3>>();
            tree3_->init(V_, F_);
        } else {
            throw std::invalid_argument("AABBTree only supports V with 2 or 3 columns.");
        }
    }

    std::tuple<Eigen::VectorXd, Eigen::VectorXi, Eigen::MatrixXd>
    squared_distance(const Eigen::MatrixXd& P) const {
        if (P.cols() != dim_) {
            throw std::invalid_argument(
                "AABBTree.squared_distance: P must have the same number of columns as V.");
        }
        Eigen::VectorXd sqrD;
        Eigen::VectorXi I;
        Eigen::MatrixXd C;
        if (dim_ == 2) {
            tree2_->squared_distance(V_, F_, P, sqrD, I, C);
        } else {
            tree3_->squared_distance(V_, F_, P, sqrD, I, C);
        }
        return std::make_tuple(sqrD, I, C);
    }

    int dim() const { return dim_; }
    const Eigen::MatrixXd& V() const { return V_; }
    const Eigen::MatrixXi& F() const { return F_; }

private:
    Eigen::MatrixXd V_;
    Eigen::MatrixXi F_;
    int dim_;
    std::unique_ptr<igl::AABB<Eigen::MatrixXd, 2>> tree2_;
    std::unique_ptr<igl::AABB<Eigen::MatrixXd, 3>> tree3_;
};

void binding_aabb_tree(py::module& m) {
    py::class_<AABBTreeWrapper>(m, "_AABBTree_cpp_impl")
        .def(py::init([](EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> F) {
                 return new AABBTreeWrapper(Eigen::MatrixXd(V), Eigen::MatrixXi(F));
             }),
             py::arg("V"), py::arg("F"))
        .def("squared_distance",
             [](const AABBTreeWrapper& self, EigenDRef<MatrixXd> P) {
                 return self.squared_distance(Eigen::MatrixXd(P));
             },
             py::arg("P"))
        .def_property_readonly("dim", &AABBTreeWrapper::dim);
}
