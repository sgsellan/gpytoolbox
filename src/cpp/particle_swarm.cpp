#include "particle_swarm.h"
#include <random>
#include <limits>
#include <iostream>
#include <algorithm>

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
    double& best_f)
{
    const int n = lb.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    // Initialize particle positions uniformly in [lb, ub].
    Eigen::MatrixXd x(n_particles, n);
    for (int i = 0; i < n_particles; ++i) {
        for (int j = 0; j < n; ++j) {
            x(i, j) = lb(j) + uniform01(gen) * (ub(j) - lb(j));
        }
    }

    Eigen::MatrixXd best_xi = x;
    Eigen::VectorXd best_fi = Eigen::VectorXd::Constant(
        n_particles, std::numeric_limits<double>::infinity());

    double current_best_f = std::numeric_limits<double>::infinity();
    Eigen::VectorXd current_best_x = x.row(0).transpose();

    for (int i = 0; i < n_particles; ++i) {
        Eigen::VectorXd xi = x.row(i).transpose();
        double f = fun(xi);
        best_xi.row(i) = xi.transpose();
        best_fi(i) = f;
        if (f < current_best_f) {
            current_best_x = xi;
            current_best_f = f;
        }
    }

    // Initialize velocities uniformly in [lb-ub, ub-lb].
    Eigen::MatrixXd v(n_particles, n);
    for (int i = 0; i < n_particles; ++i) {
        for (int j = 0; j < n; ++j) {
            const double vlb = lb(j) - ub(j);
            const double vub = ub(j) - lb(j);
            v(i, j) = vlb + uniform01(gen) * (vub - vlb);
        }
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n_particles; ++i) {
            const double rp = uniform01(gen);
            const double rg = uniform01(gen);

            Eigen::VectorXd best_neighbor;
            if (topology == "full") {
                best_neighbor = current_best_x;
            } else if (topology == "ring") {
                const int prev = (i - 1 + n_particles) % n_particles;
                const int next = (i + 1) % n_particles;
                int best_idx = i;
                double best_val = best_fi(i);
                if (best_fi(prev) < best_val) {
                    best_val = best_fi(prev);
                    best_idx = prev;
                }
                if (best_fi(next) < best_val) {
                    best_val = best_fi(next);
                    best_idx = next;
                }
                best_neighbor = best_xi.row(best_idx).transpose();
            } else {
                best_neighbor = current_best_x;
            }

            for (int j = 0; j < n; ++j) {
                v(i, j) = momentum * v(i, j)
                    + phi * rp * (best_xi(i, j) - x(i, j))
                    + phi * rg * (best_neighbor(j) - x(i, j));
                x(i, j) = x(i, j) + v(i, j);
                x(i, j) = std::max(x(i, j), lb(j));
                x(i, j) = std::min(x(i, j), ub(j));
            }

            Eigen::VectorXd xi = x.row(i).transpose();
            const double f = fun(xi);
            if (f < best_fi(i)) {
                best_xi.row(i) = xi.transpose();
                best_fi(i) = f;
                if (f < current_best_f) {
                    current_best_x = xi;
                    current_best_f = f;
                }
            }
        }
        if (verbose) {
            std::cout << "Iteration " << iter << ": f = " << current_best_f << std::endl;
        }
    }

    best_x = current_best_x;
    best_f = current_best_f;
}
