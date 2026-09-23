#ifndef TRIANGULATE_POLYGON_H
#define TRIANGULATE_POLYGON_H

#include <Eigen/Core>

void triangulate_polygon(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const double a,
    const double q,
    const bool steiner,
    Eigen::MatrixXd& V2,
    Eigen::MatrixXi& F2);

#endif
