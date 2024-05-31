#include "fine_tune_point_cloud_iter.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <numeric>
#include <random>
#include <iostream>
#include <algorithm>

#include <Eigen/Geometry>
#include <igl/signed_distance.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/fast_winding_number.h>

#include "sAABB.h"
#include "resolve_collisions_on_sphere.h"

template<int dim>
void winding_number(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& P,
    Eigen::VectorXd& W)
{
    if constexpr(dim==2) {
        // Adapted from gpytoolbox
        W.resize(P.rows());
        W.setZero();
        for(int i=0; i<P.rows(); ++i) {
            for(int j=0; j<F.rows(); ++j) {
                const Eigen::Vector2d p2vs = V.row(F(j,0)) - P.row(i),
                p2vd = V.row(F(j,1)) - P.row(i);
                W(i) -= std::atan2(p2vd(0)*p2vs(1)-p2vd(1)*p2vs(0),
                    p2vd(0)*p2vs(0)+p2vd(1)*p2vs(1)); 
            }
            W(i) /= (2. * M_PI);
        }
    } else if constexpr(dim==3) {
        igl::fast_winding_number(V, F, P, W);
    }
}


template<int dim>
void fine_tune_point_cloud_iter(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sdf_data,
    const Eigen::MatrixXd & reconstructed_V,
    const Eigen::MatrixXi & reconstructed_F,
    Eigen::MatrixXd & cloud_points,
    Eigen::MatrixXd & cloud_normals,
    Eigen::MatrixXi & feasible,
    const Eigen::VectorXi & batch,
    const int max_points_per_sphere,
    const double rng_seed,
    const int n_local_searches,
    const int local_search_iters,
    const double local_search_t,
    const double tol,
    const double clamp_value,
    const bool parallel,
    const bool verbose)

{
    using Vecd = Eigen::Matrix<double,dim,1>;
    using Veci = Eigen::Matrix<int,dim,1>;
    assert(dim==sdf_points.cols());

    const int n = sdf_points.rows();
    assert(n == sdf_data.rows());

    const bool new_points_for_infeasible_spheres = true;
    const double add_points_shrink = 0.8;
    const int line_search_iters = 20; //Alternatively, try local_search_iters.

    // We partition all indices from 1 to n into an index list for each thread.
    std::vector<int> shuffled_indices(n);
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(),
        std::minstd_rand(rng_seed));

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

    // Invert the feasible vector.
    std::vector<std::vector<int> > feasible_inv(n);
    for(int i=0; i<feasible.size(); ++i) {
        feasible_inv[feasible(i)].push_back(i);
    }

    // Compute signed distances and closest points to reconstructed mesh.
    Eigen::VectorXd d_sq, W;
    Eigen::VectorXi I;
    Eigen::MatrixXd P;
    igl::point_mesh_squared_distance(sdf_points,
        reconstructed_V, reconstructed_F,
        d_sq, I, P);
    winding_number<dim>(reconstructed_V, reconstructed_F, sdf_points, W);
    Eigen::VectorXd d_sq_cloud;
    Eigen::VectorXi I_cloud;
    Eigen::MatrixXd P_cloud;
    if(max_points_per_sphere>1) {
        // Also precompute distance from all cloud points to the reconstructed
        // mesh.
        igl::point_mesh_squared_distance(cloud_points,
            reconstructed_V, reconstructed_F, d_sq_cloud, I_cloud, P_cloud);
    }

    // Utility functions which we will use later
    const auto proj_onto_sphere = [&]
    (const Vecd& p, // Project this point
        const int i // onto this sphere
        ) -> Vecd {
        // This is not a trivial projection to the closest point on the sphere.
        // This is the projection such that the normal agrees best with the
        // surface point.
        const Vecd& c = sdf_points.row(i);
        const double s = sdf_data(i);
        const double r = std::abs(s);
        Vecd N = p - c;
        const double norm = N.norm();
        if(norm > std::numeric_limits<double>::epsilon()) {
            N /= norm;
        } else {
            N.setZero();
            N(0) = 1.;
        }
        // Here is where we depart from normal projection - we know the sign of
        // the normal from sdf_data.
        N *= (s<0 ? -1. : 1.) * (W(i)>0.5 ? -1. : 1.);

        return c + r*N;
    };
    std::vector<std::vector<int> > workspaces;
    const auto move_p_towards_q = [&]
    (const Vecd& p,  // The point to move
        const Vecd& q, // The point to move towards
        const int i, // The sphere everything should be tangent to.
        const int thread_id
        ) -> Vecd {
        const Vecd& c = sdf_points.row(i);
        const double r = std::abs(sdf_data(i));

        if((p-q).norm()<tol || r<tol || (p-c).norm()<tol) {
            return p;
        }

        // Find the great arc between p and q, and move t along it.
        // This constructs a great arc between p and q on the sphere i,
        // and theta is the angle that we move along this great arc.
        // theta is chosen so that the arc has length t, but no longer than the
        // distance between p and q.
        const Vecd x=p-c, y=q-c;
        double theta = std::min(
            //travel local_search_t along arc
            local_search_t / r,
            //the length of the entire arc.
            // std::acos(std::max(-1., std::min(1., // numerically less stable
            //     x.dot(y)/(r*r))))
            2.*std::asin(std::min(1., 0.5*(p-q).norm()/r))
            );
        for(int i=0; i<line_search_iters; ++i) {
            Vecd new_p;
            if constexpr(dim==2) {
                const double rot_sign = x(0)*y(1)-x(1)*y(0) < 0 ? -1. : 1.;
                const Eigen::Rotation2D<double> R(rot_sign*theta);
                new_p = c + R*x;
            } else if constexpr(dim==3) {
                const Vecd N = x.cross(y).normalized();
                const Eigen::AngleAxis<double> R(theta, N);
                new_p = c + R*x;
            }
            std::vector<int>& prims = workspaces[thread_id];
            aabb_tree.get_spheres_containing(new_p,
                1, -tol, //negative tol: it's ok to be inside a bit.
                i,
                prims);
            if(prims.empty()) {
                return new_p;
            }
            theta *= 0.5;
        }
        // @SILVIA: You might want to be conservative and return p here
        // in case the line search fails instead of q...
        return q;
        // return p;
    };
    const auto move_closest_p_towards_q = [&]
    (const Vecd& q, // The point to move towards
        const int i, // The sphere everything should be tangent to.
        const int thread_id
        ) -> Vecd {
        const int closest_idx = *std::min_element(
            feasible_inv[i].begin(), feasible_inv[i].end(),
            [&] (const int a, const int b) {
                //Ok to use Euclidean distance here, closest Euclidean will be
                // closest spherical.
                return (cloud_points.row(a)-q.transpose()).squaredNorm() <
                (cloud_points.row(b)-q.transpose()).squaredNorm();
            });
        const Vecd& closest_p = cloud_points.row(closest_idx);

        return move_p_towards_q(closest_p, q, i, thread_id);
    };

    // Move around points and add new ones to fine-tune.
    std::vector<std::vector<Vecd> > points_vec;
    std::vector<std::vector<int> > points_to_spheres_vec;
    std::vector<std::vector<Vecd> > old_points_on_is;
    const auto prep_move_f = [&] (const int n_threads) {
        points_vec.resize(n_threads);
        points_to_spheres_vec.resize(n_threads);
        old_points_on_is.resize(n_threads);
        workspaces.resize(n_threads);
    };

    const auto loop_move_f = [&] (const int p_i, const int thread_id) {
        // Correctly assign the loop index using our shuffled list.
        const int i = shuffled_indices[p_i];
        const Vecd& c = sdf_data.row(i);
        const double s = sdf_data(i);
        const double r = std::abs(s);
        std::vector<Vecd>& points = points_vec[thread_id];
        std::vector<int>& points_to_spheres = points_to_spheres_vec[thread_id];
        std::vector<Vecd>& old_points_on_i = old_points_on_is[thread_id];

        if(d_sq(i) < std::pow(r-tol, 2) && r>tol) {
            // The sphere intersects the surface, and the sphere is large
            // enough for an intersection to make sense.
            // We move the point around to where it actually wants to be.
            // TODO: In certain scenarios we should add new points here.
            const Vecd p_proj = proj_onto_sphere(P.row(i), i);
            if(feasible_inv[i].size()>0) {
                points.push_back(move_closest_p_towards_q(
                    p_proj, i,
                    thread_id));
                points_to_spheres.push_back(i);
                if(max_points_per_sphere>1 && feasible_inv[i].size()>0) {
                    //Sort feasible_inv so the best current points come first
                    // and have the highest chance of being kept.
                    std::sort(feasible_inv[i].begin(), feasible_inv[i].end(),
                        [&] (const int a, const int b) {
                            return d_sq_cloud(a) < d_sq_cloud(b);
                        });
                    //We added the worst offender. But now, we will also add the
                    // closest point on the reconstructed mesh for each existing
                    // point.
                    for(int j=0;
                        j<feasible_inv[i].size() && j+1<max_points_per_sphere;
                        ++j) {
                        // If the intersection is very bad add a new point,
                        // otherwise keep number of points the same.
                        if(d_sq(i) > std::pow(add_points_shrink*r, 2) &&
                            j+1>=feasible_inv[i].size()) {
                            break;
                        }
                        const Vecd p_cloud_proj = proj_onto_sphere(
                            P_cloud.row(feasible_inv[i][j]), i);
                        points.push_back(move_p_towards_q(
                            cloud_points.row(feasible_inv[i][j]),
                            p_cloud_proj, i, thread_id));
                        points_to_spheres.push_back(i);
                    }
                }
            } else if(new_points_for_infeasible_spheres) {
                points.push_back(p_proj);
                points_to_spheres.push_back(i);
            }
        } else if(d_sq(i) > std::pow(r+tol, 2) && r>tol) {
            // The closest point is outside the sphere.
            // We move the point closer to where it actually wants to be.
            const Vecd p_proj = proj_onto_sphere(P.row(i), i);
            if(feasible_inv[i].size()>0) {
                points.push_back(move_closest_p_towards_q(
                    p_proj, i,
                    thread_id));
                points_to_spheres.push_back(i);
            } else if(new_points_for_infeasible_spheres) {
                points.push_back(p_proj);
                points_to_spheres.push_back(i);
            }
        } else {
            // The sphere is tangent to the surface.
            // Keep all tangency points as they were, if they exist.
            if(feasible_inv[i].size()>0) {
                for(int j=0; j<feasible_inv[i].size(); ++j) {
                    points.push_back(cloud_points.row(feasible_inv[i][j]));
                    points_to_spheres.push_back(i);
                }
            } else if(new_points_for_infeasible_spheres) {
                const Vecd p_proj = proj_onto_sphere(P.row(i), i);
                points.push_back(p_proj);
                points_to_spheres.push_back(i);
            }
        }
    };

    std::vector<Vecd> points;
    std::vector<int> points_to_spheres;
    const auto accum_move_f = [&] (const int thread_id) {
        points.insert(points.end(),
            points_vec[thread_id].begin(),
            points_vec[thread_id].end());
        points_to_spheres.insert(points_to_spheres.end(),
            points_to_spheres_vec[thread_id].begin(),
            points_to_spheres_vec[thread_id].end());
    };

    if(parallel) {
        igl::parallel_for(n, prep_move_f, loop_move_f, accum_move_f, 1000);
    } else {
        prep_move_f(1);
        for(int i=0; i<n; ++i) {
            loop_move_f(i,0);
        }
        accum_move_f(0);
    }

    // Make the new points locally feasible if they're not already.
    Eigen::Matrix<bool, Eigen::Dynamic, 1> feasible_bool(points.size());
    feasible_bool.setConstant(false);
    std::vector<std::vector<int> > prims;
    std::vector<std::vector<Vecd> > centers;
    std::vector<std::vector<double> > radii;
    const auto prep_feasible_f = [&] (const int n_threads) {
        prims.resize(n_threads);
        centers.resize(n_threads);
        radii.resize(n_threads);
        for(int i=0; i<n_threads; ++i) {
            prims[i].reserve(n_search);
            centers[i].reserve(n_search);
            radii[i].reserve(n_search);
        }
    };

    const auto loop_feasible_f = [&] (const int i, const int thread_id) {
        //No need to shuffle again, we already shuffled the spheres.
        Vecd point = points[i];
        const int sphere_idx = points_to_spheres[i];

        // if above the clamp value, skip
        if(std::abs(sdf_data(sphere_idx)) >= clamp_value) {
            return;
        }

        for(int local_search_iter=0;
            !feasible_bool(i) && local_search_iter<=local_search_iters;
            ++local_search_iter) {
            aabb_tree.get_spheres_containing(point,
                1, -tol, //negative tol: it's ok to be inside a bit.
                sphere_idx,
                prims[thread_id]);
            if(prims[thread_id].empty()) {
                feasible_bool(i) = true;
            } else {
                if(local_search_iter>=local_search_iters) {
                    continue;
                }
                // Make locally feasible.
                aabb_tree.get_spheres_containing(point,
                    n_search, tol, //positive tol: project far enough away
                    sphere_idx,
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
                    sdf_points.row(sphere_idx), std::abs(sdf_data(sphere_idx)),
                    centers[thread_id], radii[thread_id]);
            }
        }
        points[i] = point;
    };

    const auto accum_feasible_f = [&] (const int thread_id) {
    };

    if(parallel) {
        igl::parallel_for(points.size(),
            prep_feasible_f, loop_feasible_f, accum_feasible_f, 1000);
    } else {
        prep_feasible_f(1);
        for(int i=0; i<points.size(); ++i) {
            loop_feasible_f(i,0);
        }
        accum_feasible_f(0);
    }

    // Keep only feasible points for next iteration
    const int n_feasible = feasible_bool.count();
    cloud_points.resize(n_feasible, dim);
    cloud_normals.resize(n_feasible, dim);
    feasible.resize(n_feasible, 1);
    int idx = 0;
    for(int i=0; i<points.size(); ++i) {
        const Vecd& point = points[i];
        const int sphere_idx = points_to_spheres[i];

        if(feasible_bool(i)) {
            cloud_points.row(idx) = point;
            Vecd N = point - sdf_points.row(sphere_idx).transpose();
            const double norm = N.norm();
            if(norm > std::numeric_limits<double>::epsilon()) {
                N /= norm;
            } else {
                N.setZero();
                N(0) = 1.;
            }
            cloud_normals.row(idx) = sdf_data(sphere_idx)<0 ? N : -N;
            feasible(idx) = sphere_idx;
            ++idx;
        }
    }
}


template void fine_tune_point_cloud_iter<2>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<int, -1, -1, 0, -1, -1>&, const Eigen::Matrix<int, -1, 1, 0, -1, 1>&, int, double, int, int, double, double, double, bool, bool);
template void fine_tune_point_cloud_iter<3>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<int, -1, -1, 0, -1, -1>&, const Eigen::Matrix<int, -1, 1, 0, -1, 1>&, int, double, int, int, double, double, double, bool, bool);

