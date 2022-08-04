import numpy as np
# Bindings using C++ and Eigen:
from gpytoolbox.copyleft.do_meshes_intersect import do_meshes_intersect
from gpytoolbox.copyleft.mesh_boolean import mesh_boolean
from gpytoolbox.decimate import decimate
from gpytoolbox.offset_surface import offset_surface

def lazy_cage(V,F,grid_size=50,max_iter=10,num_faces=100,force_output=True):
    """Constructs a coarse surface fully containing a given mesh

    Given a fine triangle mesh, output a coarser cage triangle mesh that fully contains the input, useful for animation and physics-based simulation.

    Parameters
    ----------
    V : numpy double array
        Matrix of vertex coordinates
    F : numpy int array
        Matrix of triangle indices
    grid_size : int, optional (default 50)
        Size of the grid on which distances are computed during cage construction (higher should give a more tightly fitting cage)
    max_iter : int, optional (default 10)
        Iterations in cage construction binary search (more gives a more tightly fitting cage)
    num_faces : int, optional (default 100)
        Desired number of faces in the cage (will be passed as an argument to decimate)
    force_output : bool, optional (default True)
        Makes the algorithm output a cage even if it intersects the input (otherwise, returns (None, None) if unsucessful)

    Returns
    -------
    U : numpy double array
        Matrix of cage mesh vertices
    G : numpy int array
        Matrix of cage triangle indices

    See Also
    --------
    decimate.

    Notes
    -----
    This construction follows the algorithm introduced by Sellán et al. in "Breaking Good: Fracture Modes for Realtime Destruction"

    Examples
    --------
    TO-DO
    """
    U = None
    G = None
    max_bb = np.max(np.abs(np.max(V,axis=0) - np.min(V,axis=0)))/2
    # print(max_bb)
    ds = np.array([0.0,max_bb])
    num_iter = 0
    while num_iter<max_iter:
        num_iter = num_iter + 1
        d = np.mean(ds)
    #   marching cubes the d-level set of unsigned distance
        vertices, faces = offset_surface(V,F.astype(np.int32),d,grid_size)
    #    decimate mesh to m faces
        decimated_vertices,decimated_faces,J,I = decimate(vertices,faces,num_faces=num_faces)

    #    "self union" to remove self-intersections
        # clean_vertices, clean_faces = mesh_union(decimated_vertices,decimated_faces.astype(np.int32),decimated_vertices,decimated_faces.astype(np.int32))
        clean_vertices, clean_faces = mesh_boolean(decimated_vertices,decimated_faces,decimated_vertices,decimated_faces,boolean_type='union')
        a,_a = do_meshes_intersect(clean_vertices,clean_faces,V,F.astype(np.int32))
        # print(_a)
        if a: # it intersects
            ds[0] = d
        else:
            ds[1] = d
            U = clean_vertices
            G = clean_faces

    if ((U is None) and (force_output)):
        U = clean_vertices
        G = clean_faces

    return U, G