# Bindings using C++ and Eigen:
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build-linux/')))
from gpytoolbox_eigen_bindings import mesh_union
from gpytoolbox_eigen_bindings import mesh_difference
from gpytoolbox_eigen_bindings import mesh_intersection
from gpytoolbox_eigen_bindings import upper_envelope
from gpytoolbox_eigen_bindings import ray_mesh_intersect
from gpytoolbox_eigen_bindings import build_octree_as_hex_mesh
from gpytoolbox_eigen_bindings import in_element_aabb
# Other stuff
from .edge_indeces import edge_indeces
from .regular_square_mesh import regular_square_mesh
from .regular_cube_mesh import regular_cube_mesh
from .linear_elasticity_stiffness import linear_elasticity_stiffness
from .linear_elasticity import linear_elasticity
from .signed_distance_polygon import signed_distance_polygon
from .metropolis_hastings import metropolis_hastings
from .ray_polyline_intersect import ray_polyline_intersect
from .poisson_surface_reconstruction import poisson_surface_reconstruction
from .fd_interpolate import fd_interpolate
from .fd_grad import fd_grad
from .fd_grad_octree import fd_grad_octree
from .fd_partial_derivative import fd_partial_derivative
from .fd_partial_derivative_octree import fd_partial_derivative_octree
from .png2poly import png2poly
from .random_points_on_polyline import random_points_on_polyline
from .normalize_points import normalize_points
from .lazy_cage import lazy_cage
from .write_ply import write_ply
from .libigl_hex_to_polyscope_hex import libigl_hex_to_polyscope_hex
from .massmatrix_octree import massmatrix_octree
from .volumes_octree import volumes_octree
from .interpolate_octree import interpolate_octree
