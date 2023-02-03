#include <npe.h>
#include <pybind11/stl.h>
#include "read_obj.h"

npe_function(_read_obj_cpp_impl)
npe_arg(file, std::string)
npe_arg(return_UV, bool)
npe_arg(return_N, bool)
npe_begin_code()
    Eigen::MatrixXd V, UV, N;
    Eigen::MatrixXi F, Ft, Fn;
    int err = read_obj(file, return_UV, return_N,
        V, F, UV, Ft, N, Fn);
    return std::make_tuple(err, V, F, UV, Ft, N, Fn);
npe_end_code()

