#include <npe.h>
#include <pybind11/stl.h>
#include <upper_envelope.h>


// // void upper_envelope(const Eigen::MatrixXd VT, const Eigen::MatrixXi FT, const Eigen::MatrixXd DT, Eigen::MatrixXd & UT, Eigen::MatrixXi & GT, Eigen::MatrixXd LT);
npe_function(upper_envelope)
npe_arg(vt, dense_double)
npe_arg(ft, dense_int)
npe_arg(dt, dense_double)
npe_begin_code()
    Eigen::MatrixXd VT(vt);
    Eigen::MatrixXi FT(ft);
    Eigen::MatrixXd DT(dt);
    Eigen::MatrixXd UT;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> LT;
    Eigen::MatrixXi GT;
    upper_envelope(VT,FT,DT,UT,GT,LT);
    return std::make_tuple(npe::move(UT),npe::move(GT),npe::move(LT));
npe_end_code()