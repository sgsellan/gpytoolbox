#ifndef REMESH_BOTSCH
#define REMESH_BOTSCH



#include <Eigen/Core>

// Primary overload with feature edges. `feature` is a list of vertex indices
// that are kept completely fixed (never moved, split, flipped or collapsed).
// `feature_edges` is a #FE by 2 list of vertex-index pairs marking feature
// edges: these edges can be split (the new midpoint becomes a feature vertex
// and the two halves become feature edges), are never flipped, and can only be
// collapsed along the feature (the merged vertex stays on a feature edge).
void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F,Eigen::VectorXd & target,int iters, Eigen::VectorXi & feature, Eigen::MatrixXi & feature_edges, bool project);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters, Eigen::VectorXi feature, Eigen::MatrixXi feature_edges, bool project);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F,Eigen::VectorXd & target,int iters, Eigen::VectorXi & feature, bool project);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters, Eigen::VectorXi feature, bool project);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F,Eigen::VectorXd & target,int iters, Eigen::VectorXi & feature);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters, bool project);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters, bool project);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F);



#endif
