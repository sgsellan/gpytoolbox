import numpy as np
import igl
from skimage.measure import marching_cubes

# Bindings using C++ and Eigen:
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/')))
from gpytoolbox_eigen_bindings import mesh_union, offset_surface, do_meshes_intersect

def lazy_cage(V,F,grid_size=50,max_iter=10,num_faces=100):

    ds = np.array([0.0,0.5])
    num_iter = 0
    while num_iter<max_iter:
        num_iter = num_iter + 1
        d = np.mean(ds)
    #   marching cubes the d-level set of unsigned distance
        vertices, faces = offset_surface(V,F.astype(np.int32),d,grid_size)
    #    decimate mesh to m faces
        flag,decimated_vertices,decimated_faces,J,I = igl.decimate(vertices,faces,num_faces)

    #    "self union" to remove self-intersections
        clean_vertices, clean_faces = mesh_union(decimated_vertices,decimated_faces.astype(np.int32),decimated_vertices,decimated_faces.astype(np.int32))
        a = do_meshes_intersect(clean_vertices,clean_faces,V,F.astype(np.int32))
        if len(a[0])>0: # it intersects
            ds[0] = d
        else:
            ds[1] = d
        U = clean_vertices
        G = clean_faces
    return clean_vertices,clean_faces