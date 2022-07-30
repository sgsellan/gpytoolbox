#ifndef REMESH_BOTSCH
#define REMESH_BOTSCH



#include <Eigen/Core>

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F,Eigen::VectorXd & target,int iters, Eigen::VectorXi & feature, bool project);


void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F,Eigen::VectorXd & target,int iters, Eigen::VectorXi & feature);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters, bool project);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters, bool project);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double);

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F);



#endif
