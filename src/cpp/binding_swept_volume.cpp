#include <npe.h>
#include <pybind11/stl.h>
#include "swept_volume/swept_volume.h"

// void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const Eigen::MatrixXd transformation_matrix, const double eps, const int num_seeds, Eigen::MatrixXd & U, Eigen::MatrixXi & G)
npe_function(_swept_volume_impl)
npe_arg(v, dense_double)
npe_arg(f, dense_int)
npe_arg(transformations, dense_double)
npe_arg(eps, double)
npe_arg(num_seeds, int)
npe_arg(verbose, bool)
npe_begin_code()
    Eigen::MatrixXd V(v);
    Eigen::MatrixXi F(f);
    Eigen::MatrixXd U;
    Eigen::MatrixXi G;
    swept_volume(V, F,transformations, eps, num_seeds, verbose, U,G);
    return std::make_tuple(npe::move(U), npe::move(G));
npe_end_code()
