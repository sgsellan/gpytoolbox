#include "octree_mc.h"

#include <Eigen/Core>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <tuple>

namespace py = pybind11;

// The Python callables get and return numpy arrays: the field maps a (k, 3)
// array of points to k values, subdivide maps the cells' (k, 3) low and high
// corners and (k,) depths to k values (nonzero: split). Both are wrapped in
// octree_marching_cubes.py so that they return float64 arrays.
void binding_octree_marching_cubes(py::module& m)
{
    m.def(
        "_octree_marching_cubes_cpp_impl",
        [](py::function sdf,
           py::function subdivide,
           const Eigen::Vector3d& lo,
           const Eigen::Vector3d& hi,
           int max_depth,
           bool binary_search,
           int binary_search_steps,
           bool psr)
        {
            omc::ScalarField f = [&](const std::vector<omc::Vec3>& points) {
                Eigen::MatrixXd P(points.size(), 3);
                for (size_t i = 0; i < points.size(); ++i)
                    for (int a = 0; a < 3; ++a) P(i, a) = points[i][a];
                const Eigen::VectorXd S = sdf(P).cast<Eigen::VectorXd>();
                return std::vector<double>(S.data(), S.data() + S.size());
            };
            omc::SubdivideFn split = [&](const std::vector<omc::Cell>& cells) {
                Eigen::MatrixXd L(cells.size(), 3), H(cells.size(), 3);
                Eigen::VectorXi depth(cells.size());
                for (size_t i = 0; i < cells.size(); ++i) {
                    for (int a = 0; a < 3; ++a) {
                        L(i, a) = cells[i].lo[a];
                        H(i, a) = cells[i].hi[a];
                    }
                    depth(i) = cells[i].depth;
                }
                const Eigen::VectorXd s = subdivide(L, H, depth).cast<Eigen::VectorXd>();
                std::vector<bool> out(s.size());
                for (Eigen::Index i = 0; i < s.size(); ++i) out[i] = s(i) != 0;
                return out;
            };

            omc::Options options;
            options.lo = {lo(0), lo(1), lo(2)};
            options.hi = {hi(0), hi(1), hi(2)};
            options.max_depth = max_depth;
            options.interpolation = binary_search ? omc::Interpolation::BinarySearch
                                                  : omc::Interpolation::Linear;
            options.binary_search_steps = binary_search_steps;
            options.method = psr ? omc::Method::PSR : omc::Method::Transvoxel;

            const omc::Mesh mesh = omc::marching_cubes(split, f, options);

            Eigen::MatrixXd V(mesh.vertices.size(), 3);
            Eigen::MatrixXi F(mesh.triangles.size(), 3);
            for (size_t i = 0; i < mesh.vertices.size(); ++i)
                for (int a = 0; a < 3; ++a) V(i, a) = mesh.vertices[i][a];
            for (size_t i = 0; i < mesh.triangles.size(); ++i)
                for (int a = 0; a < 3; ++a) F(i, a) = mesh.triangles[i][a];
            return std::make_tuple(V, F);
        },
        py::arg("sdf"),
        py::arg("subdivide"),
        py::arg("lo"),
        py::arg("hi"),
        py::arg("max_depth"),
        py::arg("binary_search"),
        py::arg("binary_search_steps"),
        py::arg("psr"));
}
