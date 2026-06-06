#ifndef COLLAPSE_EDGES
#define COLLAPSE_EDGES



#include <Eigen/Core>

// Collapses edges shorter than the lower bound. Fixed `feature` vertices are
// never collapsed. A feature edge (listed in `feature_edges`) can only be
// collapsed onto one of its endpoints, so the merged vertex stays on the
// feature. Both `feature` and `feature_edges` are remapped in place to the new
// vertex indexing after the collapses.
void collapse_edges(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature, Eigen::MatrixXi & feature_edges, Eigen::VectorXd & high, Eigen::VectorXd & low);


#endif
