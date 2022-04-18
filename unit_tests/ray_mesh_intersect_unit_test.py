import numpy as np
import igl
from context import gpytoolbox

# This tests crashes if you use the igl binding.

# This is a cube, centered at the origin, with side length 1
v,f = igl.read_triangle_mesh("unit_tests_data/cube.obj")
cam_pos = np.array([1,0.1,0.1])
cam_dir = np.array([-1,0,0])
hits = gpytoolbox.ray_mesh_intersect(cam_pos.astype(np.float64),cam_dir.astype(np.float64),v.astype(np.float64),f.astype(np.int32))
# There should only be two hits (there are three because the C++ ray_mesh_intersect doesn't work well? )
print("Number of hits:", len(hits))
# t (travelled distance) should be 0.5
t = hits[0][4]
print("Travelled distance:", t)
# intersection should be [0.5,0.1,0.1]
intersection = cam_pos + t*cam_dir
print("Intersected at: ", intersection)