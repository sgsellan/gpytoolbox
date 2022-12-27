import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Remove this with pip
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import gpytoolbox
# Generate random points on a polyline
poly = gpytoolbox.png2poly("test/unit_tests_data/illustrator.png")[0]
poly = poly - np.min(poly)
poly = poly/np.max(poly)
poly = 0.5*poly + 0.25
poly = 3*poly - 1.5
num_samples = 40
np.random.seed(2)
EC = gpytoolbox.edge_indices(poly.shape[0],closed=False)
P,I,_ = gpytoolbox.random_points_on_mesh(poly, EC, num_samples, return_indices=True)
vecs = poly[EC[:,0],:] - poly[EC[:,1],:]
vecs /= np.linalg.norm(vecs, axis=1)[:,None]
J = np.array([[0., -1.], [1., 0.]])
N = vecs @ J.T
N = N[I,:]


# Problem parameters
gs = np.array([50,50])
# Call to PSR
scalar_mean, scalar_var, grid_vertices = gpytoolbox.poisson_surface_reconstruction(P,N,gs=gs,verbose=True,stochastic=True,diagonal_probing=True)

# The probability of each grid vertex being inside the shape 
prob_out = 1 - norm.cdf(scalar_mean,0,np.sqrt(scalar_var))

gx = grid_vertices[0]
gy = grid_vertices[1]

# Plot mean and variance side by side with colormap
fig, ax = plt.subplots(1,3)
m0 = ax[0].pcolormesh(gx,gy,np.reshape(scalar_mean,gx.shape), cmap='RdBu',shading='gouraud', vmin=-np.max(np.abs(scalar_mean)), vmax=np.max(np.abs(scalar_mean)))
ax[0].scatter(P[:,0],P[:,1],30 + 0*P[:,0])
q0 = ax[0].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
ax[0].set_title('Mean')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(m0, cax=cax, orientation='vertical')

m1 = ax[1].pcolormesh(gx,gy,np.reshape(np.sqrt(scalar_var),gx.shape), cmap='plasma',shading='gouraud')
ax[1].scatter(P[:,0],P[:,1],30 + 0*P[:,0])
q1 = ax[1].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
ax[1].set_title('Variance')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(m1, cax=cax, orientation='vertical')

m2 = ax[2].pcolormesh(gx,gy,np.reshape(prob_out,gx.shape), cmap='viridis',shading='gouraud')
ax[2].scatter(P[:,0],P[:,1],30 + 0*P[:,0])
q2 = ax[2].quiver(P[:,0],P[:,1],N[:,0],N[:,1])
ax[2].set_title('Probability of being inside')
divider = make_axes_locatable(ax[2])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(m2, cax=cax, orientation='vertical')
plt.show()