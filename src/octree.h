#ifndef OCTREE_CLASS
#define OCTREE_CLASS

#include <Eigen/Core>
#include <vector>

class Octree{
    public:
        std::vector<std::vector<int>> point_indeces;
        Eigen::MatrixXi CH; // children indeces
        Eigen::MatrixXd CN; // centers
        Eigen::VectorXd W;  // length of box
        Eigen::MatrixXd all_verts; // All grid vertices
        Eigen::MatrixXi all_quads;
        Octree(Eigen::MatrixXd P);
        void get_edges(Eigen::MatrixXd & EV, Eigen::MatrixXi & EI);
};

#endif