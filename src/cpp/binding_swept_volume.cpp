#include "swept_volume/swept_volume.h"
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

void binding_swept_volume(py::module& m) {
    m.def("_swept_volume_impl",[](EigenDRef<MatrixXd> v,
                         EigenDRef<MatrixXi> f, EigenDRef<MatrixXd> transformations, double eps, int num_seeds, bool verbose)
        {
            Eigen::MatrixXd V(v);
            Eigen::MatrixXi F(f);
            Eigen::MatrixXd U;
            Eigen::MatrixXi G;
            swept_volume(V, F,transformations, eps, num_seeds, verbose, U,G);
            return std::make_tuple(U,G);
        });
    
}
// void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const Eigen::MatrixXd transformation_matrix, const double eps, const int num_seeds, Eigen::MatrixXd & U, Eigen::MatrixXi & G)
// npe_function(_swept_volume_impl)
// npe_arg(v, dense_double)
// npe_arg(f, dense_int)
// npe_arg(transformations, dense_double)
// npe_arg(eps, double)
// npe_arg(num_seeds, int)
// npe_arg(verbose, bool)
// npe_begin_code()
//     Eigen::MatrixXd V(v);
//     Eigen::MatrixXi F(f);
//     Eigen::MatrixXd U;
//     Eigen::MatrixXi G;
//     swept_volume(V, F,transformations, eps, num_seeds, verbose, U,G);
//     return std::make_tuple(npe::move(U), npe::move(G));
// npe_end_code()
