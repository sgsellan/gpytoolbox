#include <Eigen/Core>
#include "microstl.h"


int read_stl(
    const std::string& file,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F){
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
        }else if(res == microstl::Result::Success){
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
        }else{
            return -5;
        }
    };

int write_stl(
    const std::string& file,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const bool binary){
        // Write the mesh in their struct format
        microstl::Mesh mesh;
        mesh.facets.resize(F.rows());
        for (int i = 0; i < F.rows(); i++)
        {
            mesh.facets[i].v1.x = V(F(i, 0), 0);
            mesh.facets[i].v1.y = V(F(i, 0), 1);
            mesh.facets[i].v1.z = V(F(i, 0), 2);
            mesh.facets[i].v2.x = V(F(i, 1), 0);
            mesh.facets[i].v2.y = V(F(i, 1), 1);
            mesh.facets[i].v2.z = V(F(i, 1), 2);
            mesh.facets[i].v3.x = V(F(i, 2), 0);
            mesh.facets[i].v3.y = V(F(i, 2), 1);
            mesh.facets[i].v3.z = V(F(i, 2), 2);
        }
        microstl::Result res;

        if(binary){
            microstl::MeshProvider providerBinary(mesh);
            std::filesystem::path path(file);
            res = microstl::Writer::writeStlFile(path, providerBinary);
        }else{
            microstl::MeshProvider providerAscii(mesh);
		    providerAscii.ascii = true;
            std::filesystem::path path(file);
            res = microstl::Writer::writeStlFile(path, providerAscii);
        }

        if(res == microstl::Result::Success){
            return 0;
        }else if(res == microstl::Result::FileError){
            return -1;
        }else{
            return -2;
        }
    };


