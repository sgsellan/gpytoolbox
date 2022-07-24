#include <npe.h>
#include <pybind11/stl.h>
#include <in_element_aabb.h>


npe_function(in_element_aabb)
npe_arg(queries, dense_double)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_begin_code()
    Eigen::MatrixXd P(queries);
    Eigen::MatrixXd V(vt);
    Eigen::MatrixXi F(ft);
    Eigen::VectorXi I;
    in_element_aabb(P,V,F,I);
    return npe::move(I);
npe_end_code()
