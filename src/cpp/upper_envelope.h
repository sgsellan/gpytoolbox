#ifndef UPPER_ENVELOPE
#define UPPER_ENVELOPE

#include <Eigen/Core>

typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> ArrayXb;

void upper_envelope(Eigen::MatrixXd & VT, Eigen::MatrixXi & FT, Eigen::MatrixXd & DT, Eigen::MatrixXd & UT, Eigen::MatrixXi & GT, ArrayXb & LT);

#endif