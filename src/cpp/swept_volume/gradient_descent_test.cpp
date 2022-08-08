#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>

void gradient_descent_test(const std::function<double(const double)> f, const std::function<double(const double)> gf, const double x0, double & fx, double & x, std::vector<double> & intervals, std::vector<double> & values, std::vector<double> & minima){
//    for (int i = 0; i<intervals.size(); i++) {
//        std::cout << " " << intervals[i];
//    }
    //std::cout << std::endl;
    int max_iter = 100;
    //double alpha = 0.02;
    double alpha = 0.02; // CHANGE THIS BACK
    double tau,g;
    double min = 0.0;
    double max = 1.0;
    double tol = 1e-6;
    double xmin = x0;
    double xmax = x0;
    x = x0;
    
    double prev_x = 10000000.0;
    int iter = 0;
    bool stop = false;
    double x_candidate,fx_candidate;
    g = 100.0;
    //std::cout << "run" << std::endl;
    int in_existing_interval = -1;
    assert(iter<max_iter && !stop && abs(x-prev_x)>tol);
    while (iter<max_iter && !stop && abs(x-prev_x)>tol) {
        xmin = std::min(xmin,x);
        xmax = std::max(xmax,x);
        
        for (int mm = 0; mm < (intervals.size()/2); mm++) {
            if ((x >= (intervals[2*mm]-1e-6)) && (x <= (intervals[2*mm+1]+1e-6)) ){
                fx = values[mm];
                x = minima[mm];
                in_existing_interval = mm;
                break;
             //   in_existing_interval = true;
            }
        }
        if (in_existing_interval>-1) {
            break;
        }
        if (iter==0) {
            fx = f(x);
        }
//        if (fx<-0.1) {
//            break;
//        }
        
        
        
        g = gf(x);
        //std::cout << g << std::endl;
        tau = alpha;
        prev_x = x;
        for (int div = 1; div<10; div++) {
            iter = iter + 1;
            assert(iter<max_iter);
            x_candidate = x - tau* ( (double) (g > 0) - (g < 0));
            x_candidate = std::max(std::min(x_candidate,1.0),0.0);
            fx_candidate = f(x_candidate);
            if ((fx_candidate-fx)<(0.5*(x_candidate - x)*g)) {
                x = x_candidate;
                fx = fx_candidate;
                //std::cout << div << std::endl;
                break;
            }
            tau = 0.5*tau;
            if (div==9) {
                //std::cout << div << std::endl;
                stop = true;
            }
        }
    }
    
    
    if (in_existing_interval==-1) {
        // we have discovered a new interval
        intervals.push_back(xmin);
        intervals.push_back(xmax);
        values.push_back(fx);
        minima.push_back(x);
    }else{
        // grow interval
        intervals[2*in_existing_interval] = std::min(intervals[2*in_existing_interval],xmin);
        intervals[2*in_existing_interval+1] = std::max(intervals[2*in_existing_interval+1],xmax);
    }
    
  //  std::cout << iter << std::endl;
    
    
//    for (int i = 0; i<intervals.size(); i++) {
//        std::cout << " " << intervals[i];
//    }
//    std::cout << std::endl;
    
}
