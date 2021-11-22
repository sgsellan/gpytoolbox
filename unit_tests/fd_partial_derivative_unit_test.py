import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from fd_partial_derivative import fd_partial_derivative


# Choose grid size
gs = 100

# Build a grid
x, y = np.meshgrid(np.linspace(0,1,gs),np.linspace(0,1,gs))
V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)

# Build staggered grid in x direction
x, y = np.meshgrid(np.linspace(0,1,gs-1),np.linspace(0,1,gs))
Vx = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
# Build staggered grid in y direction
x, y = np.meshgrid(np.linspace(0,1,gs),np.linspace(0,1,gs-1))
Vy = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)


Dx = fd_partial_derivative(gs=gs,h=(1.0/(gs-1)),direction=0)
Dy = fd_partial_derivative(gs=gs,h=(1.0/(gs-1)),direction=1)

# all rows must sum up to zero (i.e. a constant function has zero derivative)
assert((np.isclose(Dx.sum(axis=1),np.zeros((Dx.shape[0],1)))).all())
assert((np.isclose(Dy.sum(axis=1),np.zeros((Dy.shape[0],1)))).all())

# Build linear function
f = 2*V[:,0] + 5*V[:,1]
computed_derivative_x = Dx*f
computed_derivative_y = Dy*f
# Derivatives must be 2.0 and 5.0, respectively
assert((np.isclose(computed_derivative_x,2.0*np.ones((computed_derivative_y.shape[0])))).all())
assert((np.isclose(computed_derivative_y,5.0*np.ones((computed_derivative_y.shape[0])))).all())

# Convergence test
linf_norm = 100.0
print("This experiment should print a set of decreasing values, converging")
print("towards zero and decreasing roughly by half in each iteration")
for power in range(1,13,1):
    gs = 2**power
    # Build a grid
    x, y = np.meshgrid(np.linspace(0,1,gs),np.linspace(0,1,gs))
    V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
    # Build staggered grid in x direction
    x, y = np.meshgrid(np.linspace(0,1,gs-1),np.linspace(0,1,gs))
    Vx = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
    # Build staggered grid in y direction
    x, y = np.meshgrid(np.linspace(0,1,gs),np.linspace(0,1,gs-1))
    Vy = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)
    # Build derivative matrices
    Dx = fd_partial_derivative(gs=gs,h=(1.0/(gs-1)),direction=0)
    Dy = fd_partial_derivative(gs=gs,h=(1.0/(gs-1)),direction=1)
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
    assert(linf_norm>np.max(np.abs(computed_derivative_x - fx)))
    linf_norm = np.max(np.abs(computed_derivative_x - fx))
    # Print L infinity norm of difference
    print(np.max(np.abs(computed_derivative_x - fx)))

print("Unit test passed, all asserts passed")