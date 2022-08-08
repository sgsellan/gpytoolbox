#ifndef FD_INTERPOLATE_H
#define FD_INTERPOLATE_H
#include <Eigen/Core>
#include <Eigen/Sparse>
// Construct a matrix of trilinear interpolation weights for a
// finite-difference grid at a given set of points
//
// Inputs:
//   nx  number of grid steps along the x-direction
//   ny  number of grid steps along the y-direction
//   nz  number of grid steps along the z-direction
//   h  grid step size
//   corner  list of bottom-left-front corner position of grid
//   P  n by 3 list of query point locations
// Outputs:
//   W  n by (nx*ny*nz) sparse weights matrix
//
void fd_interpolate(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  const Eigen::RowVector3d & corner,
  const Eigen::MatrixXd & P,
  Eigen::SparseMatrix<double> & W);
#endif
