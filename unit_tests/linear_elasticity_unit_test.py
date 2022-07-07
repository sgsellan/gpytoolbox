import numpy as np
from numpy.core.numeric import isclose
import polyscope as ps
from context import gpytoolbox

# Build very small mesh
V, F = gpytoolbox.regular_square_mesh(3)
V = (V + 1.)/2.
# Initial conditions
fext = 0*V
Ud0 = 0*V
U0 = V.copy()
U0[:,1] = 0.0
U0[:,0] = U0[:,0] - 0.5
# print(np.reshape(U0,(-1,1),order='F'))
dt = 0.2
ps.init()
U, sigma_v = gpytoolbox.linear_elasticity(V,F,U0,fext=fext,dt=dt,Ud0=Ud0)
Ud0 = (np.reshape(U,(-1,2),order='F') - U0)/dt
U0 = np.reshape(U,(-1,2),order='F')
# Groundtruth obtained using Matlab's linear elasticity gptoolbox function
U_groundtruth = np.array([  [-0.3660,    0.1304],
                            [0.0000,     0.1304],
                            [0.3660,     0.1304],
                            [-0.3660,   -0.0000],
                            [0.0000,    -0.0000],
                            [0.3660,     0.0000],
                            [-0.3660,   -0.1304],
                            [-0.0000,   -0.1304],
                            [0.3660,   -0.1304]])

print(np.reshape(U,(-1,2),order='F'))
print(U_groundtruth)

# Check python output matches Matlab groundtruth
assert(isclose(np.reshape(U,(-1,2),order='F'),U_groundtruth,atol=1e-4).all())

# Check another iteration, with non-zero velocity
U, sigma_v = gpytoolbox.linear_elasticity(V,F,U0,fext=fext,dt=dt,Ud0=Ud0)
Ud0 = (np.reshape(U,(-1,2),order='F') - U0)/dt
U0 = np.reshape(U,(-1,2),order='F')
# Groundtruth computed with gptoolbox's function
U_groundtruth = np.array([  [-0.2378,    0.2513],
                            [0.0000,    0.2513],
                            [0.2378,    0.2513],
                            [-0.2378,   -0.0000],
                            [0.0000,   -0.0000],
                            [0.2378,    0.0000],
                            [-0.2378,   -0.2513],
                            [-0.0000,   -0.2513],
                            [0.2378 ,  -0.2513]])
# Check python output matches Matlab groundtruth
assert(isclose(np.reshape(U,(-1,2),order='F'),U_groundtruth,atol=1e-4).all())


ps_mesh = ps.register_surface_mesh("my mesh", V + np.reshape(U,(-1,2),order='F') , F, smooth_shade=False)

# Check that over time it looks reasonable (try to make this better if Nick pushes callbacks)
for i in range(200):
    ps.show(10)
    Ud0 = (np.reshape(U,(-1,2),order='F') - U0)/dt
    U0 = np.reshape(U,(-1,2),order='F')
    U, sigma_v = gpytoolbox.linear_elasticity(V,F,U0,fext=fext,dt=dt,Ud0=Ud0)
    ps_mesh.update_vertex_positions(V + np.reshape(U,(-1,2),order='F'))

print("Unit test passed, all asserts passed")