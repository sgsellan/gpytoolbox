#include "write_obj.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <limits>


int write_obj(
    const std::string& file,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& UV,
    const Eigen::MatrixXi& Ft,
    const Eigen::MatrixXd& N,
    const Eigen::MatrixXi& Fn)
{
    bool use_default_ft = Ft.rows() == 0 && UV.rows() == V.rows();
    bool use_default_fn = Fn.rows() == 0 && N.rows() == V.rows();
    const bool write_t = Ft.rows() > 0 || use_default_ft;
    const bool write_n = Fn.rows() > 0 || use_default_fn;
    if(write_t && !use_default_ft && (Ft.rows()!=F.rows() || Ft.cols()!=F.cols())) {
        return -11; //Dimensions of Ft wrong
    }
    if(write_n && !use_default_fn && (Fn.rows()!=F.rows() || Fn.cols()!=F.cols())) {
        return -12; //Dimensions of Fn wrong
    }

    std::ofstream stream(file);
    if(!stream) {
        return -5; //File could not be opened error.
    }
    stream.precision(std::numeric_limits<double>::max_digits10);

    //Write V
    for(int i=0; i<V.rows(); ++i) {
        stream << "v ";
        for(int j=0; j<V.cols()-1; ++j) {
            stream << V(i,j) << " ";
        }
        stream << V(i,V.cols()-1) << "\n";
    }

    //Write UV
    for(int i=0; i<UV.rows(); ++i) {
        stream << "vt ";
        for(int j=0; j<UV.cols()-1; ++j) {
            stream << UV(i,j) << " ";
        }
        stream << UV(i,UV.cols()-1) << "\n";
    }

    //Write N
    for(int i=0; i<N.rows(); ++i) {
        stream << "vn ";
        for(int j=0; j<N.cols()-1; ++j) {
            stream << N(i,j) << " ";
        }
        stream << N(i,N.cols()-1) << "\n";
    }

    //Write F
    const auto write_F_element = [write_t, use_default_ft, write_n, use_default_fn, &stream, &F, &Ft, &Fn]
    (const int i, const int j) {
        stream << F(i,j) + 1;
        if(write_t || write_n) {
            stream << "/";
        }
        if(write_t) {
            if(use_default_ft) {
                stream << F(i,j) + 1;
            } else {
                stream << Ft(i,j) + 1;
            }
        }
        if(write_n) {
            if(use_default_fn) {
                stream << "/" << F(i,j) + 1;
            } else {
                stream << "/" << Fn(i,j) + 1;
            }
        }
    };
    for(int i=0; i<F.rows(); ++i) {
        stream << "f ";
        for(int j=0; j<F.cols()-1; ++j) {
            write_F_element(i,j);
            stream << " ";
        }
        write_F_element(i,F.cols()-1);
        stream << "\n";
    }

    return 0;
}

