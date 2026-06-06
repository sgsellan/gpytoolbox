#ifndef EQUALIZE_VALENCES
#define EQUALIZE_VALENCES



#include <Eigen/Core>

// Flips edges to bring vertex valences closer to 6. Feature edges (listed in
// `feature_edges`) and edges incident to a fixed `feature` vertex are never
// flipped.
void equalize_valences(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXi & feature, Eigen::MatrixXi & feature_edges);


#endif
