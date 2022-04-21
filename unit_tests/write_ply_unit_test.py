import numpy as np
import igl
from context import gpytoolbox

# To-do: evaluate this better, maybe open them with another library and see that it's consistent. I don't have time to do this right now.

v,f = igl.read_triangle_mesh("unit_tests_data/bunny.obj")
gpytoolbox.write_ply("test_file.ply",v,faces=f)
c = v[:,0]**2
gpytoolbox.write_ply("test_file_scalar.ply",v,faces=f,colors=c)
c = np.vstack(((v[:,0]**2)/np.max(v[:,0]**2),(v[:,1]**2)/np.max(v[:,1]**2),(v[:,2]**2)/np.max(v[:,0]**2))).transpose()
gpytoolbox.write_ply("test_file_rgb_normalized.ply",v,faces=f,colors=c)
c = np.round( 255*np.vstack(((v[:,0]**2)/np.max(v[:,0]**2),(v[:,1]**2)/np.max(v[:,1]**2),(v[:,2]**2)/np.max(v[:,0]**2))).transpose())
gpytoolbox.write_ply("test_file_rgb_ints.ply",v,faces=f,colors=c)

# Point cloud
gpytoolbox.write_ply("test_file_pc.ply",v)
# Point cloud with colores
gpytoolbox.write_ply("test_file_pc_rgb.ply",v,colors=c)

print("Unit test passed, all asserts passed")