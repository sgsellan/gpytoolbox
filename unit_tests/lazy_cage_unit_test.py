import numpy as np
from context import gpytoolbox
import igl
import polyscope as ps

# Load mesh
V, F = igl.read_triangle_mesh("unit_tests_data/armadillo.obj")
V = gpytoolbox.normalize_points(V)
U,G = gpytoolbox.lazy_cage(V,F,num_faces=1000,grid_size=100)

ps.init()
ps_fine_mesh = ps.register_surface_mesh("fine mesh", V, F)
ps_cage = ps.register_surface_mesh("cage", U, G)
ps_cage.set_transparency(0.5)
ps.show()