import numpy as np
from context import gpytoolbox
import igl
import polyscope as ps

# Load mesh
P = np.random.rand(5,3)
v, q = gpytoolbox.build_octree_as_hex_mesh(P)
v, SVI, SVJ, q = igl.remove_duplicate_vertices(v,q,1e-5)
t = gpytoolbox.libigl_hex_to_polyscope_hex(q)

# ps.init()
# ps_vol = ps.register_volume_mesh("test volume mesh", v, hexes=t)
# ps.show()

Dx,staggered_x = gpytoolbox.fd_partial_derivative_octree(v,t,direction=0)
Dy,staggered_y = gpytoolbox.fd_partial_derivative_octree(v,t,direction=1)
# all rows must sum up to zero (i.e. a constant function has zero derivative)
assert((np.isclose(Dx.sum(axis=1),np.zeros((Dx.shape[0],1)))).all())
assert((np.isclose(Dy.sum(axis=1),np.zeros((Dy.shape[0],1)))).all())


fun_vals = 2*v[:,0] + 5*v[:,1]
computed_derivative_x = Dx*fun_vals
computed_derivative_y = Dy*fun_vals
assert((np.isclose(computed_derivative_x,2.0*np.ones((computed_derivative_x.shape[0])))).all())
assert((np.isclose(computed_derivative_y,5.0*np.ones((computed_derivative_y.shape[0])))).all())

# Convergence test
linf_norm_x = 100.0
linf_norm_y = 100.0
linf_norm_z = 100.0
print("This experiment should print a set of decreasing values, converging")
print("towards zero")
for power in range(2,6,1):
    P = np.random.rand(2**(2*power),3)
    v, q = gpytoolbox.build_octree_as_hex_mesh(P)
    print("Octree with",str(q.shape[0]),"cells")
    v, SVI, SVJ, q = igl.remove_duplicate_vertices(v,q,1e-7)
    t = gpytoolbox.libigl_hex_to_polyscope_hex(q)

    # Build partial derivative matrices
    Dx, staggered_x = gpytoolbox.fd_partial_derivative_octree(v,t,direction=0)
    Dy, staggered_y = gpytoolbox.fd_partial_derivative_octree(v,t,direction=1)
    Dz, staggered_z = gpytoolbox.fd_partial_derivative_octree(v,t,direction=2)

    # Build non-linear function
    f = np.cos(v[:,0]) + np.sin(v[:,1]) + 3*np.cos(v[:,2])
    # Derivatives on staggered grids
    fx = -np.sin(staggered_x[:,0])
    fy =  np.cos(staggered_y[:,1])
    fz =-3*np.sin(staggered_z[:,2])
    computed_derivative_x = Dx @ f
    computed_derivative_y = Dy @ f
    computed_derivative_z = Dz @ f
    assert(linf_norm_x>np.mean(np.abs(computed_derivative_x - fx)))
    assert(linf_norm_y>np.mean(np.abs(computed_derivative_y - fy)))
    assert(linf_norm_z>np.mean(np.abs(computed_derivative_z - fz)))
    linf_norm_x = np.mean(np.abs(computed_derivative_x - fx))
    linf_norm_y = np.mean(np.abs(computed_derivative_y - fy))
    linf_norm_z = np.mean(np.abs(computed_derivative_z - fz))
    # Print L infinity norm of difference
    print(np.array([np.mean(np.abs(computed_derivative_x - fx)),np.mean(np.abs(computed_derivative_y - fy)),np.mean(np.abs(computed_derivative_z - fz))]))

print("Unit test passed, all asserts passed")