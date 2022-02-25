import numpy as np
import igl
from skimage.measure import marching_cubes

import pymesh as pm

def lazy_cage(V,F,m,grid_size=50):
#    given target # faces m
#    binary search on offset amount d
    # Make grid

    ds = np.array([0.0,0.0])
    d = np.mean(ds)
#   marching cubes the d-level set of unsigned distance
    vertices, faces, GV, side, So = igl.offset_surface(V,F,0,grid_size,0)

    
#    decimate mesh to m faces
    decimated_vertices,decimated_faces,J,I,flag = igl.decimate(vertices,faces,m)
    mesh = pm.form_mesh(decimated_vertices, decimated_faces)
    clear_mesh = pm.boolean(mesh, mesh, operation="union", engine="igl")
#    "self union" to remove self-intersections

#    if intersects input
#       increase d
#       continue
#    store as best feasible d
#    decrease d
    return V,F