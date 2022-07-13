# Bindings using C++ and Eigen:
import sys
import os
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build-linux/')))
    from gpytoolbox_eigen_bindings import mesh_union
    from gpytoolbox_eigen_bindings import mesh_difference
    from gpytoolbox_eigen_bindings import mesh_intersection
    from gpytoolbox_eigen_bindings import upper_envelope
    from gpytoolbox_eigen_bindings import ray_mesh_intersect
    from gpytoolbox_eigen_bindings import in_element_aabb
    from gpytoolbox_eigen_bindings import decimate
    from gpytoolbox_eigen_bindings import mqwf
    from gpytoolbox_eigen_bindings import remesh_botsch
    from .lazy_cage import lazy_cage
    from .linear_elasticity import linear_elasticity
except:
    print("-------------------------------------------------------------------")
    print("WARNING: You are using only the pure-python gpytoolbox functionality. Some functions will be unavailable. \n See https://github.com/sgsellan/gpytoolbox for full installation instructions.")
    print("-------------------------------------------------------------------")

# Things that do not need my bindings
# These functions require igl official bindings (and they shouldn't)
try:
    from .linear_elasticity_stiffness import linear_elasticity_stiffness
except:
    print("-------------------------------------------------------------------")
    print("WARNING: You have not installed igl bindings, and can not use linear_elasticity_stiffness.")
    print("-------------------------------------------------------------------")

# This function depends on skimage and imageio (should it?)
try:
    from .png2poly import png2poly
except:
    print("-------------------------------------------------------------------")
    print("WARNING: You have not installed the necessary libraries to use png2poly.")
    print("-------------------------------------------------------------------")


# These functions depend ONLY on numpy, scipy and each other
from .edge_indeces import edge_indeces
from .regular_square_mesh import regular_square_mesh
from .regular_cube_mesh import regular_cube_mesh
from .signed_distance_polygon import signed_distance_polygon
from .metropolis_hastings import metropolis_hastings
from .ray_polyline_intersect import ray_polyline_intersect
from .poisson_surface_reconstruction import poisson_surface_reconstruction
from .fd_interpolate import fd_interpolate
from .fd_grad import fd_grad
from .fd_partial_derivative import fd_partial_derivative
from .random_points_on_polyline import random_points_on_polyline
from .normalize_points import normalize_points
from .write_ply import write_ply
from .initialize_quadtree import initialize_quadtree
from .subdivide_quad import subdivide_quad
from .in_quadtree import in_quadtree
from .quadtree_gradient import quadtree_gradient
from .quadtree_laplacian import quadtree_laplacian
from .quadtree_boundary import quadtree_boundary
from .quadtree_children import quadtree_children
from .grad import grad
from .doublearea import doublearea
from .doublearea_intrinsic import doublearea_intrinsic
from .massmatrix import massmatrix
from .halfedges import halfedges
from .edges import edges
from .boundary_loops import boundary_loops
from .boundary_edges import boundary_edges
from .boundary_vertices import boundary_vertices
from .min_quad_with_fixed import min_quad_with_fixed
from .min_quad_with_fixed import min_quad_with_fixed_precompute
from .halfedge_lengths import halfedge_lengths
from .halfedge_lengths_squared import halfedge_lengths_squared
from .cotangent_laplacian_intrinsic import cotangent_laplacian_intrinsic
from .tip_angles import tip_angles
from .tip_angles_intrinsic import tip_angles_intrinsic
from .cotangent_laplacian import cotangent_laplacian
from .cotangent_weights_intrinsic import cotangent_weights_intrinsic
from .cotangent_weights import cotangent_weights
from .remove_duplicate_vertices import remove_duplicate_vertices
from .bad_quad_mesh_from_quadtree import bad_quad_mesh_from_quadtree
