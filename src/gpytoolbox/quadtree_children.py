import numpy as np




def quadtree_children(CH):
    # Builds a list of indeces to the child cells of the quadtree
    #
    # Inputs:
    #   CH #nodes by 4 matrix of child indeces (-1 if leaf node)
    #
    # Outputs:
    #   child_indeces list of child indeces

    return (CH[:,0]<0).nonzero()[0]