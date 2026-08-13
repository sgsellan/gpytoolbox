#pragma once

#include <Eigen/Core>

class Sphere {
public:
    double radius;                // Must be non-negative
    int sign;                     // +1 for positive, -1 for negative
    Eigen::Vector3d center;       // Position of the sphere center

    Sphere(double r, int s, const Eigen::Vector3d& c);
};