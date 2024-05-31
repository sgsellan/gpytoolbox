#include "locally_make_feasible.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <numeric>
#include <random>
#include <iostream>
#include <algorithm>
#include <limits>
#include <igl/parallel_for.h>

#include <nanoflann.hpp>

#include "sAABB.h"
#include "resolve_collisions_on_sphere.h"


template<int dim>
void locally_make_feasible(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sdf_data,
    const Eigen::MatrixXd & outside_points,
    const Eigen::VectorXi & batch,
    const int rng_seed,
    const int n_local_searches,
    const int local_search_iters,
    const double tol,
    const double clamp_value,
    const bool parallel,
    const bool verbose,
    Eigen::MatrixXd & cloud_points,
    Eigen::MatrixXd & cloud_normals,
    Eigen::MatrixXi & feasible)
{
    using Vecd = Eigen::Matrix<double,dim,1>;
    using Veci = Eigen::Matrix<int,dim,1>;
    assert(dim==sdf_points.cols());

    const int n = sdf_points.rows();
    assert(n == sdf_data.rows());

    // kd-tree to find closest outside points for each sphere
    using KdTree = nanoflann::KDTreeEigenMatrixAdaptor<
        Eigen::MatrixXd, dim, nanoflann::metric_L2_Simple, true>;
    const KdTree kd_tree(dim, std::cref(outside_points), 20);

    // AABB tree will be used to make points locally feasible.
    Eigen::MatrixXd sdf_points_batched, sdf_data_batched;
    if(batch.size()>0) {
        sdf_points_batched = sdf_points(batch, Eigen::all);
        sdf_data_batched = sdf_data(batch, 0);
    } else {
        sdf_points_batched = sdf_points;
        sdf_data_batched = sdf_data;
    }
    const int n_batched = sdf_points_batched.rows();
    assert(n_batched == sdf_data_batched.rows());
    const sAABB<dim> aabb_tree(sdf_points_batched,
        sdf_data_batched.array().abs(), 
        batch,
        tol);
    const int n_search = std::min(n_local_searches, n_batched);

    // We partition all indices from 1 to n into an index list for each thread.
    std::vector<int> shuffled_indices(n);
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(),
        std::minstd_rand(rng_seed));

    // This vector holds the points that will first be found by projection from
    // the outside, and then will be made feasible.
    std::vector<Vecd> points(n);

    // Create a feasibility vector to keep track of the feasibles.
    // In order to safely paralellize, do not use std vector bool.
    Eigen::Matrix<bool, Eigen::Dynamic, 1> feasible_bool(n);
    feasible_bool.setConstant(false);

    // We want to provide working space for the AABB tree, so the vectors are
    // not reallocated every loop iteration.
    std::vector<std::vector<int> > prims;
    std::vector<std::vector<Vecd> > centers;
    std::vector<std::vector<double> > radii;
    const auto prep_f = [&] (const int n_threads) {
        prims.resize(n_threads);
        centers.resize(n_threads);
        radii.resize(n_threads);
        for(int i=0; i<n_threads; ++i) {
            prims[i].reserve(n_search);
            centers[i].reserve(n_search);
            radii[i].reserve(n_search);
        }
    };

    const auto loop_f = [&] (const int p_i, const int thread_id) {
        // Correctly assign the loop index using our shuffled list.
        const int i = shuffled_indices[p_i];

        // if the (abs) sdf value is above the clamp, then continue, we will not add a point for this arc
        if(std::abs(sdf_data(i)) >= clamp_value) {
            feasible_bool(i) = false; // redundant, but just to be sure
            return;
        }

        // Using the kd-tree, find the closest point in the outside points
        // for each sphere assigned to this thread.
        typename KdTree::IndexType index;
        double dist;
        const Vecd u = sdf_points.row(i); //important: no reference for u here!
        const double s = sdf_data(i);
        kd_tree.index_->knnSearch(&u[0], 1, &index, &dist);

        const Vecd& p = outside_points.row(index);
        Vecd N = p - u;
        const double norm = N.norm();
        if(norm > std::numeric_limits<double>::epsilon()) {
            N /= norm;
        } else {
            N.setZero();
            N(0) = 1.;
        }
        Vecd point = u + std::abs(s)*N;

        // Using the AABB tree, make the found point locally feasible.
        for(int iter=0; !feasible_bool(i) && iter<=local_search_iters;
            ++iter) {
            aabb_tree.get_spheres_containing(point,
                1, -tol, //negative tol: it's ok to be inside a bit.
                i,
                prims[thread_id]);
            if(prims[thread_id].empty()) {
                feasible_bool(i) = true;
            } else {
                if(iter>=local_search_iters) {
                    continue;
                }
                // Make locally feasible.
                aabb_tree.get_spheres_containing(point,
                    n_search, tol, //positive tol: project far enough away
                    i,
                    prims[thread_id]);

                centers[thread_id].clear();
                radii[thread_id].clear();
                //Sort the prims to speed up sphere collision later.
                std::sort(prims[thread_id].begin(), prims[thread_id].end(),
                    [&] (const auto a, const auto b) {
                    return std::abs(sdf_data(a)) < std::abs(sdf_data(b));
                });
                for(const int prim : prims[thread_id]) {
                    centers[thread_id].push_back(sdf_points.row(prim));
                    radii[thread_id].push_back(std::abs(sdf_data(prim)));
                }
                point = resolve_collisions_on_sphere<dim>(point,
                    sdf_points.row(i), std::abs(sdf_data(i)),
                    centers[thread_id], radii[thread_id]);
            }
        }

        // Finally, write the point into our global points vector.
        points[i] = point;
    };

    const auto accum_f = [&] (const int thread_id) {
    };

    if(parallel) {
        igl::parallel_for(n, prep_f, loop_f, accum_f, 1000);
    } else {
        prep_f(1);
        for(int i=0; i<n; ++i) {
            loop_f(i,0);
        }
        accum_f(0);
    }


    // Export only feasible points
    const int n_feasible = feasible_bool.count();
    cloud_points.resize(n_feasible, dim);
    cloud_normals.resize(n_feasible, dim);
    feasible.resize(n_feasible, 1);
    int idx = 0;
    for(int i=0; i<n; ++i) {
        if(feasible_bool(i)) {
            
            cloud_points.row(idx) = points[i];
            Vecd N = points[i] - sdf_points.row(i).transpose();
            const double norm = N.norm();
            if(norm > std::numeric_limits<double>::epsilon()) {
                N /= norm;
            } else {
                N.setZero();
                N(0) = 1.;
            }
            cloud_normals.row(idx) = sdf_data(i)<0 ? N : -N;
            feasible(idx) = i;
            ++idx;
        }
    }

}


template void locally_make_feasible<2>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1> const&, int, int, int, double, double, bool, bool, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<int, -1, -1, 0, -1, -1>&);
template void locally_make_feasible<3>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, 1, 0, -1, 1> const&, int, int, int, double, double, bool, bool, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<int, -1, -1, 0, -1, -1>&);
