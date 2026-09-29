#include "particle_swarm.h"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>

using namespace Eigen;
namespace py = pybind11;
using EigenDStride = Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Ref<MatrixType, 0, EigenDStride>;

void binding_particle_swarm(py::module& m) {
    m.def("_particle_swarm_cpp_impl", [](
        const std::function<double(const Eigen::VectorXd&)>& fun,
        EigenDRef<VectorXd> lb,
        EigenDRef<VectorXd> ub,
        int n_particles,
        int max_iter,
        double momentum,
        double phi,
        bool verbose,
        const std::string& topology)
    {
        Eigen::VectorXd best_x;
        double best_f = 0.0;
        particle_swarm(fun, lb, ub, n_particles, max_iter,
                       momentum, phi, verbose, topology, best_x, best_f);
        return std::make_tuple(best_x, best_f);
    });
}
