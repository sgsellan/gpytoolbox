#include <Eigen/Core>
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include <iostream>
#include <fstream>
using namespace tinyply;
#include "example-utils.hpp"

#include <typeinfo>


// auxiliary functionality

inline std::vector<uint8_t> read_file_binary(const std::string & pathToFile)
{
    std::ifstream file(pathToFile, std::ios::binary);
    std::vector<uint8_t> fileBufferBytes;

    if (file.is_open())
    {
        file.seekg(0, std::ios::end);
        size_t sizeBytes = file.tellg();
        file.seekg(0, std::ios::beg);
        fileBufferBytes.resize(sizeBytes);
        if (file.read((char*)fileBufferBytes.data(), sizeBytes)) return fileBufferBytes;
    }
    else throw std::runtime_error("could not open binary ifstream to path " + pathToFile);
    return fileBufferBytes;
}

struct memory_buffer : public std::streambuf
{
    char * p_start {nullptr};
    char * p_end {nullptr};
    size_t size;

    memory_buffer(char const * first_elem, size_t size)
        : p_start(const_cast<char*>(first_elem)), p_end(p_start + size), size(size)
    {
        setg(p_start, p_start, p_end);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override
    {
        if (dir == std::ios_base::cur) gbump(static_cast<int>(off));
        else setg(p_start, (dir == std::ios_base::beg ? p_start : p_end) + off, p_end);
        return gptr() - p_start;
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override
    {
        return seekoff(pos, std::ios_base::beg, which);
    }
};

struct memory_stream : virtual memory_buffer, public std::istream
{
    memory_stream(char const * first_elem, size_t size)
        : memory_buffer(first_elem, size), std::istream(static_cast<std::streambuf*>(this)) {}
};

class manual_timer
{
    std::chrono::high_resolution_clock::time_point t0;
    double timestamp{ 0.0 };
public:
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    void stop() { timestamp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() * 1000.0; }
    const double & get() { return timestamp; }
};

struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct double3 { double x, y, z; };
struct rgba4 { uint8_t r, g, b, a; };
struct uint3 { uint32_t x, y, z; };
struct uint4 { uint32_t x, y, z, w; };

struct geometry
{
    std::vector<double3> vertices;
    std::vector<double3> normals;
    std::vector<float2> texcoords;
    std::vector<uint3> triangles;
    std::vector<uint4> colors;
};


// read_ply implementation

int read_ply(
    const std::string& filepath,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& N,
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& C){

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

        std::shared_ptr<PlyData> vertices, normals, colors, texcoords, faces, tripstrip;


        bool are_normals_defined = false;
        bool are_colors_defined = false;
        bool are_texcoords_defined = false;
        bool are_faces_defined = false;
        bool are_tripstrip_defined = false;


        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { }

        try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); are_normals_defined = true;}
        catch (const std::exception & e) { }

        try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" });  are_colors_defined = true;}
        catch (const std::exception & e) {  }

        try { colors = file.request_properties_from_element("vertex", { "r", "g", "b", "a" });  are_colors_defined = true;}
        catch (const std::exception & e) { }


        // Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
        // arbitrary ply files, it is best to leave this 0. 
        try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); are_faces_defined = true;}
        catch (const std::exception & e) { }


        file.read(*file_stream);


        const size_t numVerticesBytes = vertices->buffer.size_bytes();
        if (vertices->t == tinyply::Type::FLOAT32) { 
            std::vector<float3> verts(vertices->count);
            std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);
            V.resize(verts.size(), 3);
            
            for (int i = 0; i < verts.size(); i++)
            {
                // std::cout << verts[i].x << " " << verts[i].y << " " << verts[i].z << std::endl;
                V.row(i) << verts[i].x, verts[i].y, verts[i].z;
            }
        }else if(vertices->t == tinyply::Type::FLOAT64) {
            std::vector<double3> verts(vertices->count);
            std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);
            V.resize(verts.size(), 3);
            
            for (int i = 0; i < verts.size(); i++)
            {
                // std::cout << verts[i].x << " " << verts[i].y << " " << verts[i].z << std::endl;
                V.row(i) << verts[i].x, verts[i].y, verts[i].z;
            }
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
            if (normals->t == tinyply::Type::FLOAT32){
                const size_t numNormalsBytes = normals->buffer.size_bytes();
                std::vector<float3> normals_(normals->count);
                std::memcpy(normals_.data(), normals->buffer.get(), numNormalsBytes);
                N.resize(normals_.size(), 3);
                for (int i = 0; i < normals_.size(); i++)
                {
                    N.row(i) << normals_[i].x, normals_[i].y, normals_[i].z;
                }
            }else if(vertices->t == tinyply::Type::FLOAT64) {
                const size_t numNormalsBytes = normals->buffer.size_bytes();
                std::vector<double3> normals_(normals->count);
                std::memcpy(normals_.data(), normals->buffer.get(), numNormalsBytes);
                N.resize(normals_.size(), 3);
                for (int i = 0; i < normals_.size(); i++)
                {
                    N.row(i) << normals_[i].x, normals_[i].y, normals_[i].z;
                }
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

        geometry mesh;

        for (int i = 0; i < V.rows(); i++)
        {
            mesh.vertices.push_back( double3{ (double) V(i, 0),  (double) V(i, 1),  (double) V(i, 2)} );
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
                mesh.normals.push_back( double3{ (double) N(i, 0),  (double) N(i, 1),  (double) N(i, 2)} );
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



        PlyFile mesh_file;

        mesh_file.add_properties_to_element("vertex", { "x", "y", "z" }, 
            Type::FLOAT64, mesh.vertices.size(), reinterpret_cast<uint8_t*>(mesh.vertices.data()), Type::INVALID, 0);

        if(use_normals){
            mesh_file.add_properties_to_element("vertex", { "nx", "ny", "nz" },
                Type::FLOAT64, mesh.normals.size(), reinterpret_cast<uint8_t*>(mesh.normals.data()), Type::INVALID, 0);
        }

        if(use_colors){
            mesh_file.add_properties_to_element("vertex", { "red", "green", "blue", "alpha" },
                Type::UINT8, mesh.colors.size(), reinterpret_cast<uint8_t*>(mesh.colors.data()), Type::INVALID, 0);
        }


        if(use_faces){
            mesh_file.add_properties_to_element("face", { "vertex_indices" },
                Type::UINT32, mesh.triangles.size(), reinterpret_cast<uint8_t*>(mesh.triangles.data()), Type::UINT8, 3);
            }

        mesh_file.get_comments().push_back("generated by tinyply 2.3");

        // Write an ASCII or binary file
        mesh_file.write(outstream_, binary);

        return 0;
    };
