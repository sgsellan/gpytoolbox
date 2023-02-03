#include <npe.h>
#include <pybind11/stl.h>
#include "microstl/microstl_wrappers.h"


npe_function(_write_stl_cpp_impl)
npe_arg(file, std::string)
npe_arg(v, dense_double)
npe_arg(f, dense_int)
npe_arg(binary, bool)
npe_begin_code()
    // npe_Matrix_v V(v), UV(uv), N(n);
    // npe_Matrix_f F(f), Ft(ft), Fn(fn);
    // return write_obj(file, V, F, UV, Ft, N, Fn);
    return write_stl(file, v, f, binary);
npe_end_code()
//