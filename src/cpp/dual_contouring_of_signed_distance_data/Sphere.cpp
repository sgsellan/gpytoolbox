#include "Sphere.h"
#include <cassert>

Sphere::Sphere(double r, int s, const Eigen::Vector3d & c)
    : radius(r), sign(s), center(c)
{
    // Basic sanity checks
    assert(radius >= 0.0 && "Sphere radius must be non-negative.");
    assert((sign == 1 || sign == -1) && "Sphere sign must be +1 or -1.");
}