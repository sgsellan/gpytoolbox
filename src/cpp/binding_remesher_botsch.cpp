#include <npe.h>
#include <pybind11/stl.h>
#include "remesher/remesh_botsch.h"

npe_function(_remesh_botsch_cpp_impl)
npe_arg(v, dense_double)
npe_arg(f, dense_int)
npe_arg(i, int)
npe_arg(h, double)
npe_arg(feature, dense_int)
npe_arg(project, bool)
npe_begin_code()
    Eigen::MatrixXd V(v);
    Eigen::MatrixXi F(f);
    Eigen::VectorXi FEATURE(feature);
    remesh_botsch(V, F, h, i, FEATURE, project);
    return std::make_tuple(npe::move(V), npe::move(F));
npe_end_code()
