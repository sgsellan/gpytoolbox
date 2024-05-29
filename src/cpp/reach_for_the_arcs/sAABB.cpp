#include "sAABB.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <queue>
#include <random>

// Helper functions
template<int dim>
double signed_distance_to_box(const Eigen::AlignedBox<double, dim>& box,
    const Eigen::Matrix<double, dim, 1>& point)
{
    using Vec = Eigen::Matrix<double, dim, 1>;
    if(box.contains(point)) {
        double inside_distance = std::min({
            std::abs(point[0] - box.min()[0]),
            std::abs(point[1] - box.min()[1]),
            std::abs(point[2] - box.min()[2]),
            std::abs(box.max()[0] - point[0]),
            std::abs(box.max()[1] - point[1]),
            std::abs(box.max()[2] - point[2])
        });
        return -inside_distance;
    } else {
        const Vec clamped = point.cwiseMax(box.min()).cwiseMin(box.max());
        double outside_distance = (clamped - point).norm();
        return outside_distance;
    }
}

template<int dim>
double signed_distance_to_sphere(const Eigen::Matrix<double, dim, 1>& center,
    double radius,
    const Eigen::Matrix<double, dim, 1>& point)
{
    return (point - center).norm() - radius;
}


template<int dim>
sAABB<dim>::sAABB(const Eigen::MatrixXd& centers,
        const Eigen::MatrixXd& radii,
        const Eigen::MatrixXi& indices,
        const double tol)
{
    init(centers, radii, indices, tol);
}


template<int dim>
sAABB<dim>::sAABB(const Eigen::MatrixXd& centers,
        const Eigen::MatrixXd& radii,
        const double tol)
{
    Eigen::MatrixXi indices(centers.rows(), 1);
    for(int i=0; i<centers.rows(); ++i) {
        indices(i) = i;
    }
    init(centers, radii, indices, tol);
}


template<int dim>
void sAABB<dim>::init(const Eigen::MatrixXd& centers,
        const Eigen::MatrixXd& radii,
        const Eigen::MatrixXi& indices,
        const double tol)
{
    nodes.clear();

    const auto n = centers.rows();
    centers_vec.clear();
    centers_vec.reserve(n);
    radii_vec.clear();
    radii_vec.reserve(n);
    indices_vec.clear();
    indices_vec.reserve(n);
    std::vector<int> prims;
    prims.reserve(n);
    for(int i=0; i<n; ++i) {
        centers_vec.push_back(centers.row(i));
        radii_vec.push_back(radii(i));
        indices_vec.push_back(indices(i));
        prims.push_back(i);
    }
    if(centers.rows()>0) {
        nodes.push_back(Node());
        Node& node = nodes.back();
        init_node(node, centers_vec, radii_vec, prims, tol);
    }
}


template<int dim>
void sAABB<dim>::init_node(Node& node,
        const std::vector<Vec>& centers,
        const std::vector<double>& radii,
        const std::vector<int>& prims,
        const double tol)
{
    if(centers.size() == 1) {
        node.primitive = prims[0];
        node.left = nullptr;
        node.right = nullptr;

        const Vec& center = centers[0];
        const double radius = radii[0] + tol;
        node.m_box = Box(center - radius*Vec::Ones(),
            center + radius*Vec::Ones());

        return;
    }

    node.primitive = -1;

    Vec min_box, min_box_centers, max_box, max_box_centers;
    min_box << Eigen::Infinity, Eigen::Infinity, Eigen::Infinity;
    max_box << -Eigen::Infinity, -Eigen::Infinity, -Eigen::Infinity;
    min_box_centers << Eigen::Infinity, Eigen::Infinity, Eigen::Infinity;
    max_box_centers << -Eigen::Infinity, -Eigen::Infinity, -Eigen::Infinity;
    for(int i=0; i < centers.size(); ++i){ 
        const Vec& center = centers[i];
        const double radius = radii[i] + tol;
        min_box = min_box.cwiseMin(center - radius*Vec::Ones());
        max_box = max_box.cwiseMax(center + radius*Vec::Ones());
        min_box_centers = min_box_centers.cwiseMin(center);
        max_box_centers = max_box_centers.cwiseMax(center);
    }
    node.m_box = Box(min_box, max_box);

    // Now we need to split the spheres into two groups
    // We will split along the longest axis of the box
    const Box b_box_centers(min_box_centers, max_box_centers);
    const Vec box_size = node.m_box.sizes();
    int split_axis = 0;
    if(box_size(1) > box_size(0)) {
        split_axis = 1;
    }
    if(dim>2) {
        if(box_size(2) > box_size(split_axis)) {
            split_axis = 2;
        }
    }

    // To make sure that the split is not degenerate, we will sort the
    // spheres along the split axis
    std::vector<double> centers_along_axis(centers.size());
    for(int i=0; i<centers.size(); ++i) {
        centers_along_axis[i] = centers[i](split_axis);
    }
    std::sort(centers_along_axis.begin(), centers_along_axis.end());
    const double median = centers_along_axis[centers_along_axis.size()/2];

    // Now we will split the spheres into two groups
    std::vector<Vec> left_centers;
    std::vector<double> left_radii;
    std::vector<int> left_prims;
    std::vector<Vec> right_centers;
    std::vector<double> right_radii;
    std::vector<int> right_prims;
    for(int i=0; i<centers.size(); ++i) {
        // If the center along split_axis is less than the median, then it
        // goes in the left group
        const auto push_left = [&] () {
            left_centers.push_back(centers[i]);
            left_radii.push_back(radii[i]);
            left_prims.push_back(prims[i]);
        };
        const auto push_right = [&] () {
            right_centers.push_back(centers[i]);
            right_radii.push_back(radii[i]);
            right_prims.push_back(prims[i]);
        };
        if(centers[i](split_axis) < median) {
            push_left();
        } else if(centers[i](split_axis) == median) {
            // If it's equal to the median, then we put it in the group with
            // fewer spheres. This avoids degeneracies
            if(left_centers.size() < right_centers.size()) {
                push_left();
            } else {
                push_right();
            }
        } else {
            push_right();
        }
    }

    // Now we can recurse.
    nodes.push_back(Node());
    Node& left_node = nodes.back();
    node.left = &left_node;
    init_node(left_node, left_centers, left_radii, left_prims, tol);
    nodes.push_back(Node());
    Node& right_node = nodes.back();
    node.right = &right_node;
    init_node(right_node, right_centers, right_radii, right_prims, tol);

}


template<int dim>
void sAABB<dim>::get_spheres_containing(const Vec& p,
    int k,
    double tol,
    int ignore,
    // std::vector<double>& distances,
    std::vector<int>& primitives) const
{
    if(k < 1) {
        return;
    }

    // Reserve space in the vecs
    // distances.clear();
    // distances.reserve(k);
    primitives.clear();
    primitives.reserve(k);

    // BFS to find spheres, simlar to libigl's AABB.cpp
    std::queue<Node const*> queue;
    queue.push(&nodes[0]);

    while(!queue.empty()) {
        const Node& current = *queue.front();
        queue.pop();

        const int i = current.primitive;
        if(i != -1) {
            // Leaf node
            if(indices_vec[i] == ignore) {
                continue;
            }
            const double d = signed_distance_to_sphere(centers_vec[i],
                radii_vec[i], p);
            if(d < tol) {
                // distances.push_back(d);
                primitives.push_back(indices_vec[i]);
            }
            if(primitives.size()>=k) {
                return;
            }
        } else {
            // Not a leaf node
            if(current.m_box.contains(p)) {
                //Shake up which node gets put in first.
                if(queue.size()%2==0) {
                    queue.push(current.left);
                    queue.push(current.right);
                } else {
                    queue.push(current.right);
                    queue.push(current.left);
                }
            }
        }
    }
}


template<int dim>
void sAABB<dim>::get_closest_sphere(const Vec& p,
    double& distance,
    int& primitive) const
{
    const int found = get_k_closest_spheres_unsafe(1, p, &distance, &primitive);
    if(found<1) {
        distance = Eigen::Infinity;
        primitive = -1;
    }
}


template<int dim>
void sAABB<dim>::get_k_closest_spheres(int k,
    const Vec& p,
    std::vector<double>& distances,
    std::vector<int>& primitives) const
{
    if(k<1) {
        return;
    }

    distances.resize(k);
    primitives.resize(k);

    const int found = get_k_closest_spheres_unsafe(k, p,
        &distances[0], &primitives[0]);

    if(found<k) {
        distances.resize(found);
        primitives.resize(found);
    }
}


template<int dim>
int sAABB<dim>::get_k_closest_spheres_unsafe(int k,
    const Vec& p,
    double* distances,
    int* primitives) const
{
    // Min-heap to store distances and indices of closest spheres
    std::priority_queue<std::pair<double, int> > min_heap;
    const auto current_highest_in_heap = [&min_heap] () {
        return min_heap.empty() ? Eigen::Infinity : min_heap.top().first;
    };

    // Queue for AABB traversal
    using QPair = std::pair<double, Node const *>;
    std::priority_queue<QPair, std::vector<QPair>, std::greater<QPair> > queue;
    queue.emplace(0., &nodes[0]);

    while(!queue.empty()) {
        const Node& current = *queue.top().second;
        queue.pop();

        const int i = current.primitive;
        if(i != -1) { // Leaf node
            const Vec& center = centers_vec[i];
            const double radius = radii_vec[i];
            const double distance = signed_distance_to_sphere(center, radius, p);

            if(min_heap.size() < k || distance < current_highest_in_heap()) {
                min_heap.emplace(distance, i);
                if(min_heap.size() > k) {
                    min_heap.pop(); // Keep heap size at k
                }
            }
        } else { // Internal node
            const double distance = signed_distance_to_box(current.m_box, p);

            if(min_heap.size() < k || distance < current_highest_in_heap()) {
                queue.emplace(distance, current.left);
                queue.emplace(distance, current.right);
            }
        }
    }

    //Transfer from heap to vector
    int i = 0;
    const int n_found = min_heap.size();
    for(int i=0; !min_heap.empty(); ++i) {
        assert(i<k);
        distances[n_found-1-i] = min_heap.top().first;
        primitives[n_found-1-i] = indices_vec[min_heap.top().second];
        min_heap.pop();
        ++i;
    }
    return n_found;
}


//Explicit instantiation
template sAABB<2>::sAABB(const Eigen::MatrixXd&, const Eigen::MatrixXd&, const double tol);
template sAABB<3>::sAABB(const Eigen::MatrixXd&, const Eigen::MatrixXd&, const double tol);

template sAABB<2>::sAABB(const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::MatrixXi&, const double tol);
template sAABB<3>::sAABB(const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::MatrixXi&, const double tol);

template void sAABB<2>::get_spheres_containing(const Vec& p, int k, double tol, int ignore, std::vector<int>& primitives) const;
template void sAABB<3>::get_spheres_containing(const Vec& p, int k, double tol, int ignore, std::vector<int>& primitives) const;

template void sAABB<2>::get_closest_sphere(const Vec& p, double& min_distance, int& min_primitive) const;
template void sAABB<3>::get_closest_sphere(const Vec& p, double& min_distance, int& min_primitive) const;

template void sAABB<2>::get_k_closest_spheres(int k, const Vec& p, std::vector<double>& min_distances, std::vector<int>& min_primitives) const;
template void sAABB<3>::get_k_closest_spheres(int k, const Vec& p, std::vector<double>& min_distances, std::vector<int>& min_primitives) const;

template int sAABB<2>::get_k_closest_spheres_unsafe(int k, const Vec& p, double* min_distances, int* min_primitives) const;
template int sAABB<3>::get_k_closest_spheres_unsafe(int k, const Vec& p, double* min_distances, int* min_primitives) const;

