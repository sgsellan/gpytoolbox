#include <igl/doublearea.h>
#include <igl/per_face_normals.h>


void random_points_on_mesh(const int n, const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, Eigen::MatrixXd & X,Eigen::MatrixXd & N){
    // not optimized, could be logarithmic but I'm too tired
    X.resize(n,3);
    Eigen::VectorXd C,A;
    igl::doublearea(V,F,A);
    igl::per_face_normals(V,F,N);
    C.resize(F.rows());
    
    double AX = 0;
    for(int j = 0; j < F.rows(); j++){
        AX = AX + A(j);
        C(j) = AX;
    }
    C = C/AX;
    
    for (int i = 0; i<n; i++) {
        double rn = (float) std::rand()/RAND_MAX;
        double a1 = (float) std::rand()/RAND_MAX;
        double a2 = (float) std::rand()/RAND_MAX;
        int ff = 0;
        for (int facenum = 0; facenum < F.rows(); facenum++) {
            if (C(facenum)>rn) {
                
                ff = facenum;
                N.row(i) = N.row(ff);
                break;
            }
        }
        //std::cout << rn << std::endl;
        if (a1+a2 > 1) {
            a1 = 1 - a1;
            a2 = 1 - a2;
        }
        X.row(i) = V.row(F(ff,0)) + a1*(V.row(F(ff,1))-V.row(F(ff,0))) + a2*(V.row(F(ff,2))-V.row(F(ff,0)));
        
    }
    
}

