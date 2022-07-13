import numpy as np

def volume(v,t):
    # Computes a vector containing the volumes of all tetrahedra in a mesh
    #
    # Inputs:
    #       v #v by dim numpy array of point position coordinates
    #       t #t by  4  numpy array of tetrahedra indeces into v 
    #
    # Output:
    #       u #v by dim numpy array of output point position coordinates
    #
    
    # Dimension:
    a = v[t[:,0],:]
    b = v[t[:,1],:]
    c = v[t[:,2],:]
    d = v[t[:,3],:]

    vols = -np.sum(np.multiply(a-d,np.cross(b-c,c-d,axis=1)),axis=1)/6.

    return vols