#include <Eigen/Core>
#include "microstl/microstl.h"
// #include "read_stl.h"

int read_stl( const std::string& file, Eigen::MatrixXd& V, Eigen::MatrixXi& F){
        microstl::MeshReaderHandler meshHandler;
        auto res = microstl::Reader::readStlFile(file, meshHandler);
        if(res == microstl::Result::LineLimitError){
            return -1;
        }else if(res == microstl::Result::ParserError){
            return -2;
        }else if(res == microstl::Result::MissingDataError){
            return -3;
        }else if(res == microstl::Result::FileError){
            return -4;
        }
        V.resize(3*meshHandler.mesh.facets.size(), 3);
        F.resize(meshHandler.mesh.facets.size(), 3);
        int ind = 0;
        for (const microstl::Facet& facet : meshHandler.mesh.facets)
	    {
            V.row(ind*3 + 0) << facet.v1.x, facet.v1.y, facet.v1.z;
            V.row(ind*3 + 1) << facet.v2.x, facet.v2.y, facet.v2.z;
            V.row(ind*3 + 2) << facet.v3.x, facet.v3.y, facet.v3.z;
            F.row(ind) << ind*3 + 0, ind*3 + 1, ind*3 + 2;
            ind = ind + 1;
	    }
        return 0;
    };