import numpy as np
from context import gpytoolbox


# Choose grid size
gs = np.array([19,15])
h = 1.0/(gs-1)

# Build a grid
x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]))
V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)

# Build staggered grid in x direction
x, y = np.meshgrid(np.linspace(0,1,gs[0]-1),np.linspace(0,1,gs[1]))
Vx = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
# Build staggered grid in y direction
x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]-1))
Vy = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)

# Build partial derivative matrices
Dx = gpytoolbox.fd_partial_derivative(gs=gs,h=h,direction=0)
Dy = gpytoolbox.fd_partial_derivative(gs=gs,h=h,direction=1)

# all rows must sum up to zero (i.e. a constant function has zero derivative)
assert((np.isclose(Dx.sum(axis=1),np.zeros((Dx.shape[0],1)))).all())
assert((np.isclose(Dy.sum(axis=1),np.zeros((Dy.shape[0],1)))).all())

# Build linear function
f = 2*V[:,0] + 5*V[:,1]
computed_derivative_x = Dx*f
computed_derivative_y = Dy*f
# Derivatives must be 2.0 and 5.0, respectively
assert((np.isclose(computed_derivative_x,2.0*np.ones((computed_derivative_x.shape[0])))).all())
assert((np.isclose(computed_derivative_y,5.0*np.ones((computed_derivative_y.shape[0])))).all())

# Convergence test
linf_norm_x = 100.0
linf_norm_y = 100.0
print("This experiment should print a set of decreasing values, converging")
print("towards zero and decreasing roughly by half in each iteration")
for power in range(3,13,1):
    gs = np.array([2**power,2**power - 2])
    h = 1.0/(gs-1)
    # Build a grid
    x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]))
    V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)

    # Build staggered grid in x direction
    x, y = np.meshgrid(np.linspace(0,1,gs[0]-1),np.linspace(0,1,gs[1]))
    Vx = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
    # Build staggered grid in y direction
    x, y = np.meshgrid(np.linspace(0,1,gs[0]),np.linspace(0,1,gs[1]-1))
    Vy = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)

    # Build partial derivative matrices
    Dx = gpytoolbox.fd_partial_derivative(gs=gs,h=h,direction=0)
    Dy = gpytoolbox.fd_partial_derivative(gs=gs,h=h,direction=1)
    # Build non-linear function
    f = np.cos(V[:,0]) + np.sin(V[:,1])
    # Derivatives on staggered grids
    fx = -np.sin(Vx[:,0])
    fy =  np.cos(Vy[:,1])
    # Computed derivatives using our matrices
    computed_derivative_x = Dx*f
    computed_derivative_y = Dy*f
    # Print L infinity norm of difference
    # Make sure norm is decreasing
    assert(linf_norm_x>np.max(np.abs(computed_derivative_x - fx)))
    assert(linf_norm_y>np.max(np.abs(computed_derivative_y - fy)))
    linf_norm_x = np.max(np.abs(computed_derivative_x - fx))
    linf_norm_y = np.max(np.abs(computed_derivative_y - fy))
    # Print L infinity norm of difference
    print(np.array([np.max(np.abs(computed_derivative_x - fx)),np.max(np.abs(computed_derivative_y - fy))]))

print("Unit test passed, all asserts passed")