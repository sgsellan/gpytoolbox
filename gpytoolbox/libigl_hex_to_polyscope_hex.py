import numpy as np

def libigl_hex_to_polyscope_hex(Q):
    # Generates a regular tetrahedral mesh of a one by one by one cube. 
    # Each grid cube is decomposed into 6 reflectionally symmetric tetrahedra
    #
    # Input:
    #       Q a #Q by 8 matrix of indexed hexes a-la-libigl octree
    # Output:
    #       T a #Q by 8 matrix of indexed hexes a-la-polyscope

    correspondence = [0,1,3,2,6,7,5,4]
    T = Q.copy()
    for i in range(len(correspondence)):
        T[:,i] = Q[:,correspondence[i]]

    return T
