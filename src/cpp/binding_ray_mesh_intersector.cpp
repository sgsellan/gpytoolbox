#include <igl/embree/EmbreeIntersector.h>
#include <igl/Hit.h>
#include <igl/parallel_for.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <memory>
#include <limits>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;

// Wrapper that owns V, F and a persistent igl::embree::EmbreeIntersector so
// repeated ray-mesh intersection queries against the same mesh skip the
// Embree-scene construction cost.
class RayMeshIntersectorWrapper {
public:
    RayMeshIntersectorWrapper(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
        : V_(V), F_(F) {
        ei_.init(V_.cast<float>(), F_, true);
    }

    std::tuple<Eigen::VectorXd, Eigen::VectorXi, Eigen::MatrixXd>
    intersect(const Eigen::MatrixXd& sources, const Eigen::MatrixXd& dirs) const {
        const int n = sources.rows();
        Eigen::VectorXd ts(n);
        Eigen::VectorXi ids(n);
        Eigen::MatrixXd lambdas(n, 3);
        igl::parallel_for(n, [&](const int si) {
            Eigen::Vector3f s = sources.row(si).cast<float>();
            Eigen::Vector3f d = dirs.row(si).cast<float>();
            igl::Hit<float> hit;
            const float tnear = 1e-4f;
            if (ei_.intersectRay(s, d, hit, tnear)) {
                ids(si) = hit.id;
                ts(si) = hit.t;
                lambdas(si, 0) = 1.0 - hit.u - hit.v;
                lambdas(si, 1) = hit.u;
                lambdas(si, 2) = hit.v;
            } else {
                ids(si) = -1;
                ts(si) = std::numeric_limits<float>::infinity();
                lambdas.row(si).setZero();
            }
        });
        return std::make_tuple(ts, ids, lambdas);
    }

private:
    Eigen::MatrixXd V_;
    Eigen::MatrixXi F_;
    igl::embree::EmbreeIntersector ei_;
};

void binding_ray_mesh_intersector(py::module& m) {
    py::class_<RayMeshIntersectorWrapper>(m, "_RayMeshIntersector_cpp_impl")
        .def(py::init([](EigenDRef<MatrixXd> V, EigenDRef<MatrixXi> F) {
                 return new RayMeshIntersectorWrapper(Eigen::MatrixXd(V), Eigen::MatrixXi(F));
             }),
             py::arg("V"), py::arg("F"))
        .def("intersect",
             [](const RayMeshIntersectorWrapper& self,
                EigenDRef<MatrixXd> sources, EigenDRef<MatrixXd> dirs) {
                 return self.intersect(Eigen::MatrixXd(sources), Eigen::MatrixXd(dirs));
             },
             py::arg("sources"), py::arg("dirs"));
}
