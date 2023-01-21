#include <npe.h>
#include <pybind11/stl.h>
#include "microstl/microstl_wrappers.h"

npe_function(_read_stl_cpp_impl)
npe_arg(file, std::string)
npe_begin_code()
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    int s = read_stl(file, V, F);
    return std::make_tuple(s, npe::move(V), npe::move(F));
npe_end_code()

