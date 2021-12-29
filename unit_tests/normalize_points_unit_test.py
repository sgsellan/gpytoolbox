import numpy as np
import igl
from context import gpytoolbox


# Test: 3D Mesh taken from the internet is centered at 0
v,f = igl.read_triangle_mesh("unit_tests_data/bunny.obj")
u = gpytoolbox.normalize_points(v)
# Check that it's centered at zero: max and min should be symmetric
assert(np.all(np.min(u,axis=0)==-np.max(u,axis=0)))
# Also it shouldn't escape the [-0.5,0.5] intervals
assert(np.min(u)==-0.5)
assert(np.max(u)==0.5)

center = np.array([0.5,0.5,0.5])
u = gpytoolbox.normalize_points(v,center)
# Should be fully contained in [0,1] intervals
assert(np.min(u)==0.0)
assert(np.max(u)==1.0)
# Check that it's centered at 0.5: max and min should be symmetric
assert(np.all(np.max(u,axis=0)==1.0-np.min(u,axis=0)))

print("Unit test passed, all asserts passed")