#ifndef SPLIT_EDGES_UNTIL_BOUND
#define SPLIT_EDGES_UNTIL_BOUND



#include <Eigen/Core>

void split_edges_until_bound(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature, Eigen::MatrixXi & feature_edges, Eigen::VectorXd & high, Eigen::VectorXd & low);


#endif
