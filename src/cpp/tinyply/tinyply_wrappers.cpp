#include <Eigen/Core>
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include <iostream>
#include <fstream>
using namespace tinyply;


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


struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct double3 { double x, y, z; };
struct uint3 { uint32_t x, y, z; };
struct uint4 { uint32_t x, y, z, w; };

int read_ply(
    const std::string& filepath,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& N,
    Eigen::MatrixXd& C){
        std::unique_ptr<std::istream> file_stream;
        std::vector<uint8_t> byte_buffer;

        bool preload_into_memory = false;

        if (preload_into_memory)
        {
            byte_buffer = read_file_binary(filepath);
            file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
        }
        else
        {
            file_stream.reset(new std::ifstream(filepath, std::ios::binary));
        }

        if (!file_stream || file_stream->fail()) return -1;

        tinyply::PlyFile file;
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

        bool are_vertices = true;
        bool are_faces = true;
        bool are_normals = true;
        bool are_colors = true;


        std::shared_ptr<PlyData> vertices, normals, colors, texcoords, faces, tripstrip;

        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { are_vertices = false; }

        try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
        catch (const std::exception & e) { are_faces = false; }

        try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
        catch (const std::exception & e) { are_normals = false; }

        try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }); are_colors = true; }
        catch (const std::exception & e) { are_colors = false; }

        try { colors = file.request_properties_from_element("vertex", { "r", "g", "b", "a" }); are_colors = true; }
        catch (const std::exception & e) { are_colors = false; }

        file.read(*file_stream);

        const size_t numVerticesBytes = vertices->buffer.size_bytes();
        std::vector<float3> verts(vertices->count);
        std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);

        

        

        

        

        V.resize(verts.size(), 3);
        
        for (int i = 0; i < verts.size(); i++)
        {
            // std::cout << verts[i].x << " " << verts[i].y << " " << verts[i].z << std::endl;
            V.row(i) << verts[i].x, verts[i].y, verts[i].z;
        }

        if (are_faces){
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


        if(are_normals){
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
        
        if(are_colors){
            const size_t numColorsBytes = colors->buffer.size_bytes();
            std::vector<uint4> colors_(colors->count);
            std::memcpy(colors_.data(), colors->buffer.get(), numColorsBytes);
            C.resize(colors_.size(), 4);
            for (int i = 0; i < colors_.size(); i++)
            {
                C.row(i) << colors_[i].x, colors_[i].y, colors_[i].z, colors_[i].w;
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
    const Eigen::MatrixXd& C,
    const bool binary){

        if(binary){
            std::filebuf fb_binary;
            fb_binary.open(filename, std::ios::out | std::ios::binary);
            std::ostream outstream_(&fb_binary);
            if (outstream_.fail()) throw std::runtime_error("failed to open " + filename);
        }else{
            std::filebuf fb_ascii;
            fb_ascii.open(filename, std::ios::out);
            std::ostream outstream_(&fb_ascii);
            if (outstream_.fail()) throw std::runtime_error("failed to open " + filename);
        }

        tinyply::PlyFile file;
        std::vector<double3> verts;
        
        
        
        // Populate the vectors...
        for (int i = 0; i < V.rows(); i++)
        {
            verts.push_back(double3{V(i, 0), V(i, 1), V(i, 2)});
        }

        file.add_properties_to_element("vertex", { "x", "y", "z" }, 
        Type::FLOAT32, file.vertices.size(), reinterpret_cast<uint8_t*>(file.verts.data()), Type::INVALID, 0);

        if (F.rows() > 0){
            std::vector<uint3> faces;
            for (int i = 0; i < F.rows(); i++)
            {
                faces.push_back(uint3{F(i, 0), F(i, 1), F(i, 2)});
            }
            file.add_properties_to_element("face", { "vertex_indices" }, 
            Type::UINT32, file.faces.size(), reinterpret_cast<uint8_t*>(file.faces.data()), Type::UINT8, 3);
        }

        if (N.rows() > 0){
            std::vector<double3> normals;
            for (int i = 0; i < N.rows(); i++)
            {
                normals.push_back(double3{N(i, 0), N(i, 1), N(i, 2)});
            }
            file.add_properties_to_element("vertex", { "nx", "ny", "nz" }, 
            Type::FLOAT32, file.normals.size(), reinterpret_cast<uint8_t*>(file.normals.data()), Type::INVALID, 0);
        }

        if (C.rows() > 0){
            std::vector<uint4> colors;
            for (int i = 0; i < C.rows(); i++)
            {
                colors.push_back(uint4{C(i, 0), C(i, 1), C(i, 2), C(i, 3)});
            }
            file.add_properties_to_element("vertex", { "red", "green", "blue", "alpha" }, 
            Type::UINT8, file.colors.size(), reinterpret_cast<uint8_t*>(file.colors.data()), Type::INVALID, 0);
        }
        
        file.get_comments().push_back("generated by tinyply 2.3");


        file.write(outstream_, false);

        return 0;
    };
