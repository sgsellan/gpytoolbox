#ifndef PARTICLE_SWARM_H
#define PARTICLE_SWARM_H

#include <Eigen/Core>
#include <functional>
#include <string>

void particle_swarm(
    const std::function<double(const Eigen::VectorXd&)>& fun,
    const Eigen::VectorXd& lb,
    const Eigen::VectorXd& ub,
    int n_particles,
    int max_iter,
    double momentum,
    double phi,
    bool verbose,
    const std::string& topology,
    Eigen::VectorXd& best_x,
    double& best_f);

#endif
