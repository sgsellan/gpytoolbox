#ifndef gradient_descent
#define gradient_descent
#include <Eigen/Core>
#include <iostream>
#include <vector>

void gradient_descent_test(const std::function<double(const double)> f, const std::function<double(const double)> gf, const double x0, double & fx, double & x, std::vector<double> & intervals, std::vector<double> & values, std::vector<double> & minima);
#endif
