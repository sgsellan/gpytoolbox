import numpy as np


def doublearea_intrinsic(l_sq,F):
    # Construct the doublearea of each element of a triangle mesh using only
    # intrinsic information (squared halfedge edge lengths).
    #
    # Input:
    #       l_sq  #F by 3 numpy array of squared halfedge lengths as computed
    #             by halfedge_lengths_squared
    #       F  #F by 3 int numpy array of face/edge vertex indices into V
    #
    # Output:
    #       A  #F vector of twice the (unsigned) area 

    assert F.shape == l_sq.shape
    assert F.shape[1]==3
    assert np.all(l_sq >= 0)

    l = np.sqrt(l_sq)

    # Using Kahan's formula
    # https://people.eecs.berkeley.edu/~wkahan/Triangle.pdf
    a,b,c = l[:,0], l[:,1], l[:,2]
    dblA = 0.5 * np.sqrt((a+(b+c)) * (c-(a-b)) * (c+(a-b)) * (a+(b-c)))

    return dblA
