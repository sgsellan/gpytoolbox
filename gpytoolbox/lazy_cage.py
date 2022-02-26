import numpy as np
import igl
from skimage.measure import marching_cubes

# Bindings using C++ and Eigen:
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/')))
from gpytoolbox_eigen_bindings import mesh_union, offset_surface

#import pymesh as pm

def lazy_cage(V,F,m,grid_size=50):
#    given target # faces m
#    binary search on offset amount d
    # Make grid

    ds = np.array([0.0,0.0])
    d = np.mean(ds)
#   marching cubes the d-level set of unsigned distance
    vertices, faces, GV, side, So = offset_surface(V,F,0.1,grid_size,0)

    
#    decimate mesh to m faces
    decimated_vertices,decimated_faces,J,I,flag = igl.decimate(vertices,faces,m)
#    "self union" to remove self-intersections
    clean_vertices, clean_faces = mesh_union(vertices,faces.astype(np.int32),vertices,faces.astype(np.int32))
#    if intersects input
#       increase d
#       continue
#    store as best feasible d
#    decrease d
    return clean_vertices,clean_faces