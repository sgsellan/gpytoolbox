#include <npe.h>
#include <pybind11/stl.h>
#include "in_element_aabb.h"


npe_function(_in_element_aabb_cpp_impl)
npe_arg(queries, dense_double)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_begin_code()
    Eigen::VectorXi I;
    in_element_aabb(queries,vt,ft,I);
    return npe::move(I);
npe_end_code()
