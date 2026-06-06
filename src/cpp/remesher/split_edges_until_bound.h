#ifndef SPLIT_EDGES_UNTIL_BOUND
#define SPLIT_EDGES_UNTIL_BOUND



#include <Eigen/Core>

// Splits all (manifold) edges longer than the upper bound. Edges incident to a
// fixed `feature` vertex are left untouched (they freeze their one-ring, as in
// the original algorithm). Edges listed in `feature_edges` *are* split: when a
// feature edge is split, the new midpoint becomes a feature vertex and the two
// halves are added to `feature_edges` (which is updated in place).
void split_edges_until_bound(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature, Eigen::MatrixXi & feature_edges, Eigen::VectorXd & high, Eigen::VectorXd & low);


#endif
