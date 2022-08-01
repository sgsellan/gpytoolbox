import numpy as np
# Bindings using C++ and Eigen:
import sys
import os
from gpytoolbox.do_meshes_intersect import do_meshes_intersect
from gpytoolbox.mesh_boolean import mesh_boolean
from gpytoolbox.decimate import decimate
from gpytoolbox.offset_surface import offset_surface

def lazy_cage(V,F,grid_size=50,max_iter=10,num_faces=100):

    ds = np.array([0.0,0.5])
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
        a,_ = do_meshes_intersect(clean_vertices,clean_faces,V,F.astype(np.int32))
        if a: # it intersects
            ds[0] = d
        else:
            ds[1] = d
        U = clean_vertices
        G = clean_faces
    return clean_vertices,clean_faces