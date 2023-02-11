#include <Eigen/Core>
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include <iostream>
#include <fstream>
using namespace tinyply;
#include "example-utils.hpp"

#include <typeinfo>

int read_ply(
    const std::string& filepath,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& N,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& C){
        // std::cout << "........................................................................\n";
        // std::cout << "Now Reading: " << filepath << std::endl;

        std::unique_ptr<std::istream> file_stream;
        std::vector<uint8_t> byte_buffer;


        bool preload_into_memory = false;
        // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a 
        // stream is a net win for parsing speed, about 40% faster. 
        if (preload_into_memory)
        {
            byte_buffer = read_file_binary(filepath);
            file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
        }
        else
        {
            file_stream.reset(new std::ifstream(filepath, std::ios::binary));
        }

        if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + filepath);

        file_stream->seekg(0, std::ios::end);
        const float size_mb = file_stream->tellg() * float(1e-6);
        file_stream->seekg(0, std::ios::beg);

        PlyFile file;
        file.parse_header(*file_stream);

        // std::cout << "\t[ply_header] Type: " << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
        // for (const auto & c : file.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
        // for (const auto & c : file.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl;

        // for (const auto & e : file.get_elements())
        // {
        //     std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
        //     for (const auto & p : e.properties)
        //     {
        //         std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
        //         if (p.isList) std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
        //         std::cout << std::endl;
        //     }
        // }

        // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers. 
        // See examples below on how to marry your own application-specific data structures with this one. 
        std::shared_ptr<PlyData> vertices, normals, colors, texcoords, faces, tripstrip;


        bool are_normals_defined = false;
        bool are_colors_defined = false;
        bool are_texcoords_defined = false;
        bool are_faces_defined = false;
        bool are_tripstrip_defined = false;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties 
        // like vertex position are hard-coded: 
        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { }

        try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); are_normals_defined = true;}
        catch (const std::exception & e) { }

        try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" });  are_colors_defined = true;}
        catch (const std::exception & e) {  }

        try { colors = file.request_properties_from_element("vertex", { "r", "g", "b", "a" });  are_colors_defined = true;}
        catch (const std::exception & e) { }

        try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); are_texcoords_defined = true;}
        catch (const std::exception & e) { }

        // Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
        // arbitrary ply files, it is best to leave this 0. 
        try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); are_faces_defined = true;}
        catch (const std::exception & e) { }

        // Tristrips must always be read with a 0 list size hint (unless you know exactly how many elements
        // are specifically in the file, which is unlikely); 
        try { tripstrip = file.request_properties_from_element("tristrips", { "vertex_indices" }, 0); are_tripstrip_defined = true;}
        catch (const std::exception & e) {}

        manual_timer read_timer;

        read_timer.start();
        file.read(*file_stream);
        read_timer.stop();

        const float parsing_time = static_cast<float>(read_timer.get()) / 1000.f;
        // std::cout << "\tparsing " << size_mb << "mb in " << parsing_time << " seconds [" << (size_mb / parsing_time) << " MBps]" << std::endl;

        // if (vertices)   std::cout << "\tRead " << vertices->count  << " total vertices "<< std::endl;
        // if (normals)    std::cout << "\tRead " << normals->count   << " total vertex normals " << std::endl;
        // if (colors)     std::cout << "\tRead " << colors->count << " total vertex colors " << std::endl;
        // if (texcoords)  std::cout << "\tRead " << texcoords->count << " total vertex texcoords " << std::endl;
        // if (faces)      std::cout << "\tRead " << faces->count     << " total faces (triangles) " << std::endl;
        // if (tripstrip)  std::cout << "\tRead " << (tripstrip->buffer.size_bytes() / tinyply::PropertyTable[tripstrip->t].stride) << " total indicies (tristrip) " << std::endl;


        const size_t numVerticesBytes = vertices->buffer.size_bytes();
        std::vector<float3> verts(vertices->count);
        std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);

        

        

        

        

        V.resize(verts.size(), 3);
        
        for (int i = 0; i < verts.size(); i++)
        {
            // std::cout << verts[i].x << " " << verts[i].y << " " << verts[i].z << std::endl;
            V.row(i) << verts[i].x, verts[i].y, verts[i].z;
        }

        if (are_faces_defined){
            const size_t numFacesBytes = faces->buffer.size_bytes();
            std::vector<uint3> ply_faces(faces->count);
            std::memcpy(ply_faces.data(), faces->buffer.get(), numFacesBytes);
            F.resize(ply_faces.size(), 3);
            for (int i = 0; i < ply_faces.size(); i++)
            {
                F.row(i) << ply_faces[i].x, ply_faces[i].y, ply_faces[i].z;
            }
        }else{
            F.resize(0, 0);
        }


        if(are_normals_defined){
            const size_t numNormalsBytes = normals->buffer.size_bytes();
            std::vector<float3> normals_(normals->count);
            std::memcpy(normals_.data(), normals->buffer.get(), numNormalsBytes);
            N.resize(normals_.size(), 3);
            for (int i = 0; i < normals_.size(); i++)
            {
                N.row(i) << normals_[i].x, normals_[i].y, normals_[i].z;
            }
        }else{
            N.resize(0, 0);
        }
        
        if(are_colors_defined){
            const size_t numColorsBytes = colors->buffer.size_bytes();
            std::vector<rgba4> colors_(colors->count);
            std::memcpy(colors_.data(), colors->buffer.get(), numColorsBytes);
            C.resize(colors_.size(), 4);
            for (int i = 0; i < colors_.size(); i++)
            {
                C.row(i) << colors_[i].r, colors_[i].g, colors_[i].b, colors_[i].a;
            }
        }else{
            C.resize(0, 0);
        }


        return 0;

    };

int write_ply(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& N,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& C,
    const bool binary){

        // geometry cube = make_cube_geometry();

        geometry mesh;

        for (int i = 0; i < V.rows(); i++)
        {
            mesh.vertices.push_back( float3{ (float) V(i, 0),  (float) V(i, 1),  (float) V(i, 2)} );
        }

        bool use_faces = false;
        if(F.rows() > 0){
            use_faces = true;
            for (int i = 0; i < F.rows(); i++)
            {
                mesh.triangles.push_back( uint3{ (uint32_t) F(i, 0),  (uint32_t) F(i, 1),  (uint32_t) F(i, 2)} );
            }
        }


        bool use_normals = false;
        if(N.rows() > 0){
            use_normals = true;
            for (int i = 0; i < N.rows(); i++)
            {
                mesh.normals.push_back( float3{ (float) N(i, 0),  (float) N(i, 1),  (float) N(i, 2)} );
            }
        }


        bool use_colors = false;
        if(C.rows() > 0){
            use_colors = true;
            for (int i = 0; i < C.rows(); i++)
            {
                mesh.colors.push_back( uint4{ (uint8_t) C(i, 0),  (uint8_t) C(i, 1),  (uint8_t) C(i, 2), (uint8_t) C(i, 3)} );
            }
        }
        // cube = mesh;
        

        std::filebuf fb;

        // binary = false;
        // std::cout << "binary: " << binary << std::endl;

        if(binary){
            fb.open(filename, std::ios::out | std::ios::binary);
        }else{
            fb.open(filename, std::ios::out);
        }

        std::ostream outstream_(&fb);

        tinyply::PlyFile file;



        PlyFile cube_file;

        cube_file.add_properties_to_element("vertex", { "x", "y", "z" }, 
            Type::FLOAT32, mesh.vertices.size(), reinterpret_cast<uint8_t*>(mesh.vertices.data()), Type::INVALID, 0);

        if(use_normals){
            cube_file.add_properties_to_element("vertex", { "nx", "ny", "nz" },
                Type::FLOAT32, mesh.normals.size(), reinterpret_cast<uint8_t*>(mesh.normals.data()), Type::INVALID, 0);
        }

        if(use_colors){
            cube_file.add_properties_to_element("vertex", { "red", "green", "blue", "alpha" },
                Type::UINT8, mesh.colors.size(), reinterpret_cast<uint8_t*>(mesh.colors.data()), Type::INVALID, 0);
        }
        // cube_file.add_properties_to_element("vertex", { "u", "v" },
        //     Type::FLOAT32, cube.texcoords.size() , reinterpret_cast<uint8_t*>(cube.texcoords.data()), Type::INVALID, 0);

        if(use_faces){
            cube_file.add_properties_to_element("face", { "vertex_indices" },
                Type::UINT32, mesh.triangles.size(), reinterpret_cast<uint8_t*>(mesh.triangles.data()), Type::UINT8, 3);
            }

        cube_file.get_comments().push_back("generated by tinyply 2.3");

        // Write an ASCII file
        cube_file.write(outstream_, binary);









        // std::vector<float3> verts;
        
        
        
        // // Populate the vectors...
        // for (int i = 0; i < V.rows(); i++)
        // {
        //     verts.push_back(float3{ (float) V(i, 0), (float) V(i, 1), (float) V(i, 2)});
        // }

        // file.add_properties_to_element("vertex", { "x", "y", "z" }, 
        // Type::FLOAT32, verts.size(), reinterpret_cast<uint8_t*>(verts.data()), Type::INVALID, 0);

        

        // // if (N.rows() > 0){
        // //     std::cout << "Writing normals" << std::endl;
        // //     std::vector<double3> normals;
        // //     for (int i = 0; i < N.rows(); i++)
        // //     {
        // //         // print for debugging
        // //         normals.push_back(double3{N(i, 0), N(i, 1), N(i, 2)});
        // //         std::cout << normals[i].x << " " << normals[i].y << " " << normals[i].z << std::endl;
        // //     }
        // //     file.add_properties_to_element("vertex", { "nx", "ny", "nz" }, 
        // //     Type::FLOAT64, normals.size(), reinterpret_cast<uint8_t*>(normals.data()), Type::INVALID, 0);
        // // }

        // if (F.rows() > 0){
        //     std::cout << "Writing faces" << std::endl;
        //     std::vector<uint3> faces;
        //     for (int i = 0; i < F.rows(); i++)
        //     {
        //         uint3 face = { (uint32_t) F(i, 0), (uint32_t) F(i, 1), (uint32_t) F(i, 2)};
        //         faces.push_back(face);
        //         // std::cout << faces[i].x << " " << faces[i].y << " " << faces[i].z << std::endl;
        //     }
        //     // print all entries in faces
        //     for (int i = 0; i < faces.size(); i++)
        //     {
        //         std::cout << faces[i].x << " " << faces[i].y << " " << faces[i].z << std::endl;
        //     }
        //     std::cout << typeid(faces).name() << '\n';
        //     file.add_properties_to_element("face", { "vertex_indices" }, 
        //     Type::UINT32, faces.size(), reinterpret_cast<uint8_t*>(faces.data()), Type::UINT8, 3);
        // }

        // // if (C.rows() > 0){
        // //     std::vector<uint4> colors;
        // //     for (int i = 0; i < C.rows(); i++)
        // //     {
        // //         colors.push_back(uint4{(uint32_t) C(i, 0), (uint32_t) C(i, 1), (uint32_t) C(i, 2), (uint32_t) C(i, 3)});
        // //     }
        // //     file.add_properties_to_element("vertex", { "red", "green", "blue", "alpha" }, 
        // //     Type::UINT8, colors.size(), reinterpret_cast<uint8_t*>(colors.data()), Type::INVALID, 0);
        // // }
        
        // file.get_comments().push_back("generated by tinyply 2.3");


        // file.write(outstream_, false);

        return 0;
    };
