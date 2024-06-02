#ifndef sAABB_H
#define sAABB_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <deque>


template<int dim>
class sAABB
{
public:
    using Vec = Eigen::Matrix<double, dim, 1>;
    using Box = Eigen::AlignedBox<double, dim>;

    struct Node
    {
        Node *left = nullptr;
        Node *right = nullptr;
        int primitive = -1;

        Box m_box;
    };

    // The tol parameter creates a safety buffer around each sphere.
    sAABB(const Eigen::MatrixXd& centers,
        const Eigen::MatrixXd& radii,
        const double tol);

    // The tol parameter creates a safety buffer around each sphere.
    sAABB(const Eigen::MatrixXd& centers,
        const Eigen::MatrixXd& radii,
        const Eigen::MatrixXi& indices,
        const double tol);

    // Get the closest sphere to p.
    // It has index min_primitive and distance min_distance
    void get_closest_sphere(const Vec& p,
        double& distance,
        int& primitive) const;

    // Get the k closest spheres to p.
    // It has index min_primitive and distance min_distance
    void get_k_closest_spheres(int k,
        const Vec& p,
        std::vector<double>& distances,
        std::vector<int>& primitives) const;

    // Get all spheres that p is inside (up to k).
    // tol determines how much outside/inside the sphere is still considered
    //  actually inside for the final inside/outside test, but not tree traversal.
    // Ignore the primitive ignore (set to -1 to not ignore anything).
    // Make sure to pre-reserve the space in distances and primitives
    void get_spheres_containing(const Vec& p,
        int k,
        double tol,
        int ignore,
        // std::vector<double>& distances,
        std::vector<int>& primitives) const;

private:
    std::vector<Vec> centers_vec;
    std::vector<double> radii_vec;
    std::vector<int> indices_vec;

    // This has to be a deque, so pointers aren't invalidated on size change.
    std::deque<Node> nodes;

    // The tol parameter creates a safety buffer around each sphere.
    void init(const Eigen::MatrixXd& centers,
        const Eigen::MatrixXd& radii,
        const Eigen::MatrixXi& prims,
        const double tol);
    void init_node(Node& node,
        const std::vector<Vec>& centers,
        const std::vector<double>& radii,
        const std::vector<int>& indices,
        const double tol);

    int get_k_closest_spheres_unsafe(int k,
                                      const Vec& p,
                                      double* distances,
                                      int* primitives) const;
};

#endif