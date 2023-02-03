#include "read_obj.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <string>

// Return an std::tuple with the string iterators to beginning and end of first
// whitespace-delimited substring in [begin, end]
auto non_space(
    const std::string::iterator begin,
    const std::string::iterator end)
{
    auto ret_begin = end;
    auto ret_end = end;
    for(auto it=begin; it!=end; ++it) {
        if(*it!=' ' && *it!='\t') {
            ret_begin = it;
            break;
        }
    }
    if(ret_begin!=end){
        for(auto it=ret_begin+1; it!=end; ++it) {
            if(*it==' ' || *it=='\t') {
                ret_end = it;
                break;
            }
        }
    }
    return std::make_tuple(ret_begin, ret_end);
}

// Determine how many whitespace-delimited tokens there are in this line
int count_non_spaces(
    const std::string::iterator begin,
    const std::string::iterator end)
{
    int cnt=0;
    auto itb = begin;
    auto ite = begin;
    while(itb!=end) {
        //Find next token. If it exists, count up.
        std::tie(itb,ite) = non_space(itb,end);
        if(itb!=ite) {
            ++cnt;
            itb=ite;
        }
    }
    return cnt;
}

// Turn tupled iterators into double, NAN on error.
double tup_it_to_double(
    const std::tuple<std::string::iterator,std::string::iterator> t,
    std::string& work)
{
    work.assign(std::get<0>(t), std::get<1>(t));
    try {
        return std::stod(work);
    }
    catch(std::invalid_argument e) {
        return NAN;
    }
}

// Turn tupled token into int and subtract one, -1 on error.
int it_to_int(
    std::string::iterator begin,
    std::string::iterator end,
    std::string& work)
{
    work.assign(begin, end);
    try {
        return std::stoi(work) - 1;
    }
    catch(std::invalid_argument e) {
        return -1;
    }
}

int read_obj(
    const std::string& file,
    const bool return_UV,
    const bool return_N,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& UV,
    Eigen::MatrixXi& Ft,
    Eigen::MatrixXd& N,
    Eigen::MatrixXi& Fn)
{

    std::cout << "Using CPP read_obj" << std::endl;
    //Reset the output arrays and implement smart growing/shrinking strategy.
    V.resize(0,0);
    int n = 0;
    const auto initialize_V = [&V,&n] (const int cols) {
        V.conservativeResize(1,cols);
        n = 1;
    };
    const auto addrow_V = [&V,&n] () {
        ++n;
        if(n>V.rows()) {
            V.conservativeResize(2*V.rows(), Eigen::NoChange);
        }
    };
    UV.resize(0,0);
    int nt = 0;
    const auto initialize_UV = [&UV,&nt] (const int cols) {
        UV.conservativeResize(1,cols);
        nt = 1;
    };
    const auto addrow_UV = [&UV,&nt] () {
        ++nt;
        if(nt>UV.rows()) {
            UV.conservativeResize(2*UV.rows(), Eigen::NoChange);
        }
    };
    N.resize(0,0);
    int nn = 0;
    const auto initialize_N = [&N,&nn] (const int cols) {
        N.conservativeResize(1,cols);
        nn = 1;
    };
    const auto addrow_N = [&N,&nn] () {
        ++nn;
        if(nn>N.rows()) {
            N.conservativeResize(2*N.rows(), Eigen::NoChange);
        }
    };
    F.resize(0,0);
    Ft.resize(0,0);
    Fn.resize(0,0);
    int m = 0;
    const auto initialize_F = [&F,&Ft,&Fn,&m,return_UV,return_N]
    (const int cols) {
        F.conservativeResize(1,cols);
        if(return_UV) {
            Ft.conservativeResize(1,cols);
            Ft.array() = -1;
        }
        if(return_N) {
            Fn.conservativeResize(1,cols);
            Fn.array() = -1;
        }
        m = 1;
    };
    const auto addrow_Fs = [&F,&Ft,&Fn,&m,return_UV,return_N] () {
        ++m;
        if(m>F.rows()) {
            F.conservativeResize(2*F.rows(), Eigen::NoChange);
            if(return_UV) {
                int oldrows = Ft.rows();
                Ft.conservativeResize(F.rows(), Eigen::NoChange);
                Ft.bottomRows(F.rows()-oldrows).array() = -1;
            }
            if(return_N) {
                int oldrows = Fn.rows();
                Fn.conservativeResize(F.rows(), Eigen::NoChange);
                Fn.bottomRows(F.rows()-oldrows).array() = -1;
            }
        }
    };

    std::ifstream stream(file);
    if(!stream) {
        return -5; //File could not be opened error.
    }
    stream.imbue(std::locale("en_US.UTF-8"));

    std::string line, work;
    while(std::getline(stream,line)) {
        int idx=0;
        if(line.size()==0) {
            continue;
        }
        switch(line[0]) {
            case '#':
            {
                //Comment
                break;
            }
            case 'v':
            {
                if(line.size()<2) {
                    return -7; //Ill-formed line.
                }
                switch(line[1]) {
                    case ' ':
                    {
                        //Vertex coord
                        if(V.size()==0) {
                            //The V array has not yet been initialized.
                            initialize_V
                            (count_non_spaces(line.begin()+1, line.end()));
                        } else {
                            addrow_V();
                        }
                        std::string::iterator at = line.begin()+1;
                        for(int i=0; i<V.cols(); ++i) {
                            const auto its = non_space(at, line.end());
                            V(n-1,i) = tup_it_to_double(its, work);
                            if(std::get<1>(its) != line.end()) {
                                at = std::get<1>(its)+1;
                            }
                        }
                        break;
                    }
                    case 't':
                    {
                        //Texture coord
                        if(!return_UV) {
                            break;
                        }
                        if(UV.size()==0) {
                            //The UV array has not yet been initialized.
                            initialize_UV
                            (count_non_spaces(line.begin()+2, line.end()));
                        } else {
                           addrow_UV();
                        }
                        std::string::iterator at = line.begin()+2;
                        for(int i=0; i<UV.cols(); ++i) {
                            const auto its = non_space(at, line.end());
                            UV(nt-1,i) = tup_it_to_double(its, work);
                            if(std::get<1>(its) != line.end()) {
                                at = std::get<1>(its)+1;
                            }
                        }
                        break;
                    }
                    case 'n':
                    {
                        //Normal coord
                        if(!return_N) {
                            break;
                        }
                        if(N.size()==0) {
                            //The N array has not yet been initialized.
                            initialize_N
                            (count_non_spaces(line.begin()+2, line.end()));
                        } else {
                            addrow_N();
                        }
                        std::string::iterator at = line.begin()+2;
                        for(int i=0; i<N.cols(); ++i) {
                            const auto its = non_space(at, line.end());
                            N(nn-1,i) = tup_it_to_double(its, work);
                            if(std::get<1>(its) != line.end()) {
                                at = std::get<1>(its)+1;
                            }
                        }
                        break;
                    }
                    default:
                    {
                        //Unrecognized start to line, ignore.
                        break;
                    }
                }
                break;
            }
            case 'f':
            {
                if(line.size()<2) {
                    return -7; //Ill-formed line.
                }
                if(line[1] != ' ') {
                    //Unrecognized start to line, ignore.
                    break;
                }
                if(F.size()==0) {
                    //The F array has not yet been initialized.
                    //Our job here is to find out i
                    const int k = count_non_spaces(line.begin()+1, line.end());
                    if(k != 3) {
                        return -8; //Only triangle meshes currently supported.
                    }
                    initialize_F(k);
                } else {
                    addrow_Fs();
                }
                std::string::iterator at = line.begin()+1;
                for(int i=0; i<F.cols(); ++i) {
                    const auto its = non_space(at, line.end());
                    //Split again, this time by forward slashes.
                    const auto slash1 = std::find
                    (std::get<0>(its), std::get<1>(its), '/');
                    F(m-1,i) = it_to_int(std::get<0>(its), slash1, work);
                    if(slash1 != std::get<1>(its)) {
                        const auto slash2 = std::find
                        (slash1+1, std::get<1>(its), '/');
                        if(return_UV && slash1+1!=slash2) {
                            Ft(m-1,i) = it_to_int(slash1+1, slash2, work);
                        }
                        if(return_N &&
                            slash2!=std::get<1>(its) &&
                            slash2+1!=std::get<1>(its)) {
                            Fn(m-1,i) = it_to_int(slash2+1, std::get<1>(its), work);
                        }
                    }
                    if(std::get<1>(its) != line.end()) {
                        at = std::get<1>(its)+1;
                    }
                }
                break;
            }
            default:
            {
                //Unrecognized start to line, ignore.
                break;
            }
        }
    }

    //Shrink arrays to size
    if(V.rows()>n) {
        V.conservativeResize(n, Eigen::NoChange);
    }
    if(UV.rows()>nt) {
        UV.conservativeResize(nt, Eigen::NoChange);
    }
    if(N.rows()>nn) {
        N.conservativeResize(nn, Eigen::NoChange);
    }
    if(F.rows()>m) {
        F.conservativeResize(m, Eigen::NoChange);
    }
    if(Ft.rows()>m) {
        Ft.conservativeResize(m, Eigen::NoChange);
    }
    if(Fn.rows()>m) {
        Fn.conservativeResize(m, Eigen::NoChange);
    }

    return 0;
}

