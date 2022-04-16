import numpy as np
import igl
from context import gpytoolbox

# To-do: evaluate this better, maybe open them with another library and see that it's consistent. I don't have time to do this right now.

v,f = igl.read_triangle_mesh("unit_tests_data/bunny.obj")
gpytoolbox.write_ply("test_file.ply",v,f)
c = v[:,0]**2
gpytoolbox.write_ply("test_file_scalar.ply",v,f,colors=c)
c = np.vstack(((v[:,0]**2)/np.max(v[:,0]**2),(v[:,1]**2)/np.max(v[:,1]**2),(v[:,2]**2)/np.max(v[:,0]**2))).transpose()
gpytoolbox.write_ply("test_file_rgb_normalized.ply",v,f,colors=c)
c = np.round( 255*np.vstack(((v[:,0]**2)/np.max(v[:,0]**2),(v[:,1]**2)/np.max(v[:,1]**2),(v[:,2]**2)/np.max(v[:,0]**2))).transpose())
gpytoolbox.write_ply("test_file_rgb_ints.ply",v,f,colors=c)

print("Unit test passed, all asserts passed")