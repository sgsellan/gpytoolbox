#include "point_cloud_to_mesh.h"

#include <random>
#include <iostream>
#include <algorithm>
#include <limits>

#include <PreProcessor.h>
#include <Reconstructors.h>

template<typename Real, int dim>
static Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>
sphere(const int i) {
    return Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>();
}

template<typename Real, unsigned int dim>
struct PointNormalSampler : public Reconstructor::InputSampleStream<Real, dim>
{
    using Mat = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    PointNormalSampler(
        const Mat& _cloud_points,
        const Mat& _cloud_normals) :
    cloud_points(_cloud_points),
    cloud_normals(_cloud_normals)
    {}

    void reset()
    {
        idx = 0;
    }

    bool base_read(Point<Real, dim> &p, Point<Real, dim> &n)
    {
        if(idx >= cloud_points.rows()) {
            return false;
        }
        for(int i=0; i<dim; ++i) {
            p[i] = cloud_points(idx,i);
            n[i] = cloud_normals(idx,i);
        }
        ++idx;
        return true;
    }

protected:
    const Mat& cloud_points;
    const Mat& cloud_normals;
    Eigen::Index idx = 0;
};


template<typename Real, unsigned int dim>
struct VertexWriter : public Reconstructor::OutputVertexStream<Real, dim>
{
    using Vec = Eigen::Matrix<Real, dim, 1>;

    VertexWriter(std::vector<Vec>& _vertices) : vertices(_vertices)
    {}

    void base_write(Point<Real, dim> p, Point<Real, dim>, Real)
    {
        Vec x;
        for(int i=0; i<dim; ++i) {
            x[i] = p[i];
        }
        vertices.push_back(x);
    }
protected:
    std::vector<Vec> &vertices;
};


template<typename Int, unsigned int dim>
struct FaceWriter : public Reconstructor::OutputFaceStream<dim-1>
{
    using Vec = Eigen::Matrix<Int, dim, 1>;

    FaceWriter(std::vector<Vec>& _faces) : faces(_faces)
    {}

    void base_write(const std::pair<node_index_type, node_index_type> &polygon)
    {
        Vec x;
        x << polygon.first, polygon.second;
        faces.push_back(x);
    }

    void base_write(const std::vector<node_index_type> &polygon)
    {
        Vec x;
        for(int i=0; i<dim; ++i) {
            x[i] = polygon[i];
        }
        faces.push_back(x);
    }
protected:
    std::vector<Vec> &faces;
};


template<typename Real, typename Int,
PointCloudReconstructionOuterBoundaryType outerBoundaryType,
unsigned int dim>
void point_cloud_to_mesh(
    const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>& _cloud_points,
    const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>& _cloud_normals,
    const Real screening_weight,
    const int depth,
    const bool parallel,
    const bool verbose,
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>& V,
    Eigen::Matrix<Int, Eigen::Dynamic, Eigen::Dynamic>& F)
{
    using Vecd = Eigen::Matrix<Real, dim, 1>;
    using Veci = Eigen::Matrix<Int, dim, 1>;
    using Matd = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    //Use the Kazhdan PSR code.
    //Following Reconstruction.example.cpp and PoissonRecon.cpp

    if(parallel) {
#ifdef _OPENMP
        ThreadPool::Init( ThreadPool::OPEN_MP , std::thread::hardware_concurrency() );
#else // !_OPENMP
        ThreadPool::Init( ThreadPool::THREAD_POOL , std::thread::hardware_concurrency() );
#endif // _OPENMP
    }

    Reconstructor::Poisson::SolutionParameters<Real> solver_params;
    solver_params.verbose = verbose;
    solver_params.depth = static_cast<unsigned int>(depth);
    solver_params.pointWeight = screening_weight;

    // Do scaling
    const bool oded_scaling = true;
    Vecd trans;
    trans.setZero();
    Real scale = 1.;
    Matd cloud_points, cloud_normals;
    if(oded_scaling) {
        solver_params.scale = -1.;
        Vecd mins = _cloud_points.colwise().minCoeff();
        Vecd maxs = _cloud_points.colwise().maxCoeff();
        const Vecd tdif = (maxs - mins).array() + 1e-6;
        mins -= 1.1 * tdif;
        maxs += 1.1 * tdif;
        mins.array() -= 1e-6;
        maxs.array() += 1e-6;
        trans = mins;
        scale = (maxs - mins).maxCoeff();
        cloud_points = (_cloud_points.rowwise() - trans.transpose()) / scale;
        cloud_normals = _cloud_normals;
    } else {
        solver_params.scale = 1.1;
        cloud_points = _cloud_points;
        cloud_normals = _cloud_normals;
    }

    Reconstructor::LevelSetExtractionParameters extraction_params;
    extraction_params.forceManifold = true;
    extraction_params.polygonMesh = false;
    extraction_params.verbose = verbose;

    constexpr BoundaryType obt =
    outerBoundaryType==PointCloudReconstructionOuterBoundaryType::Dirichlet ?
    BOUNDARY_DIRICHLET : BOUNDARY_NEUMANN;
    constexpr auto FEMSig = FEMDegreeAndBType<
    Reconstructor::Poisson::DefaultFEMDegree, obt>::Signature;

    PointNormalSampler<Real, dim> pn_sampler(cloud_points, cloud_normals);
    using Implicit = Reconstructor::Poisson::Implicit<Real, dim, FEMSig>;
    Implicit implicit(pn_sampler, solver_params, nullptr);

    std::vector<Vecd> vertices;
    std::vector<Veci> faces;
    VertexWriter<Real, dim> vertex_writer(vertices);
    FaceWriter<Int, dim> face_writer(faces);
    implicit.extractLevelSet(vertex_writer, face_writer, extraction_params);

    V.resize(vertices.size(), dim);
    for(size_t i=0; i<vertices.size(); ++i) {
        V.row(i) = vertices[i] * scale + trans;
    }
    F.resize(faces.size(), dim);
    for(size_t i=0; i<faces.size(); ++i) {
        F.row(i) = faces[i];
    }

    if(parallel) {
        ThreadPool::Terminate();
    }
}


template void point_cloud_to_mesh<double, int, (PointCloudReconstructionOuterBoundaryType)0, 2u>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, double, int, bool, bool, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<int, -1, -1, 0, -1, -1>&);
template void point_cloud_to_mesh<double, int, (PointCloudReconstructionOuterBoundaryType)0, 3u>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, double, int, bool, bool, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<int, -1, -1, 0, -1, -1>&);
template void point_cloud_to_mesh<double, int, (PointCloudReconstructionOuterBoundaryType)1, 2u>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, double, int, bool, bool, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<int, -1, -1, 0, -1, -1>&);
template void point_cloud_to_mesh<double, int, (PointCloudReconstructionOuterBoundaryType)1, 3u>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, double, int, bool, bool, Eigen::Matrix<double, -1, -1, 0, -1, -1>&, Eigen::Matrix<int, -1, -1, 0, -1, -1>&);


