#include "fd_interpolate.h"

void fd_interpolate(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  const Eigen::RowVector3d & corner,
  const Eigen::MatrixXd & P,
  Eigen::SparseMatrix<double> & W)
{
  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  //
std::vector<Eigen::Triplet<double>> ijv;
W.resize(P.rows(),nx*ny*nz);
  for(int pp = 0; pp < P.rows(); pp = pp+1){
  	int i = std::floor((P(pp,0) - corner(0))/h);
  	int j = std::floor((P(pp,1) - corner(1))/h);
  	int k = std::floor((P(pp,2) - corner(2))/h);
  	double xd = (P(pp,0) - (h*i + corner(0)))/h;
  	double yd = (P(pp,1) - (h*j + corner(1)))/h;
  	double zd = (P(pp,2) - (h*k + corner(2)))/h;
	int ind = i + nx*(j + k * ny);
	ijv.emplace_back(pp,ind,(1-xd)*(1-yd)*(1-zd));
	ijv.emplace_back(pp,ind+1,xd*(1-yd)*(1-zd));
	ijv.emplace_back(pp,ind+nx,(1-xd)*yd*(1-zd));
	ijv.emplace_back(pp,ind+nx+1,xd*yd*(1-zd));
	ijv.emplace_back(pp,ind+nx*ny,(1-xd)*(1-yd)*zd);
	ijv.emplace_back(pp,ind+1+nx*ny,xd*(1-yd)*zd);
	ijv.emplace_back(pp,ind+nx+nx*ny,(1-xd)*yd*zd);
	ijv.emplace_back(pp,ind+nx+1+nx*ny,xd*yd*zd);
  }
  W.setFromTriplets(ijv.begin(),ijv.end());
  ////////////////////////////////////////////////////////////////////////////
}
