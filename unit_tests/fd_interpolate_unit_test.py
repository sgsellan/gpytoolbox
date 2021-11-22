import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from fd_interpolate import fd_interpolate
from regular_square_mesh import regular_square_mesh


# Choose grid size
gs = 13
V, F = regular_square_mesh(gs)

# Random set of points
P = np.random.rand(10,2)

# Random grid corner
corner = np.random.rand(2)

# Displace by corner
P = P + np.tile(corner,(P.shape[0],1))
V = V + np.tile(corner,(V.shape[0],1))

W = fd_interpolate(P,gs=gs,h=(1.0/(gs-1)),corner=corner)
# all rows must sum up to one
assert((np.isclose(W.sum(axis=1),np.ones((W.shape[0],1)))).all())

# Does W do what it says it does? Can it interpolate the grid positions?
assert((np.isclose(P,W @ V)).all())

# Choose a linear function f = 3*x + 5*y
# Bilinear interpolation should exactly compute this
fP = 3.0*P[:,0] + 5.0*P[:,1]
fgrid = 3.0*V[:,0] + 5.0*V[:,1]
finterp = W @ fgrid
assert((np.isclose(fP,finterp)).all())
