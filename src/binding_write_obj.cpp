#include <npe.h>
#include <pybind11/stl.h>
#include <write_obj.h>


npe_function(_write_obj_cpp_impl)
npe_arg(file, std::string)
npe_arg(v, dense_double)
npe_arg(f, dense_int)
npe_arg(uv, dense_double)
npe_arg(ft, dense_int)
npe_arg(n, dense_double)
npe_arg(fn, dense_int)
npe_begin_code()
    Eigen::MatrixXd V(v), UV(uv), N(n);
    Eigen::MatrixXi F(f), Ft(ft), Fn(fn);
    return write_obj(file, V, F, UV, Ft, N, Fn);
npe_end_code()
// ///