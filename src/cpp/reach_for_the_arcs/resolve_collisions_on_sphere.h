#ifndef RESOLVE_COLLISIONS_ON_SPHERE_H
#define RESOLVE_COLLISIONS_ON_SPHERE_H

#include <Eigen/Core>

template<int dim>
Eigen::Matrix<double,dim,1> resolve_collisions_on_sphere(
    //The point we move around to resolve
    const Eigen::Matrix<double,dim,1>& p,
    //p must remain on this sphere
    const Eigen::Matrix<double,dim,1>& c, const double r,
    //The spheres we resolve collisions with
    const std::vector<Eigen::Matrix<double,dim,1> >& d,
    const std::vector<double>& s);

#endif