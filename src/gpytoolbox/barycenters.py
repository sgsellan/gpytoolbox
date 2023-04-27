import numpy as np



def barycenters(V,F):
    """TO-DO
    """     

    B = np.zeros((F.shape[0],V.shape[1]))
    for i in range(F.shape[1]):
        B += V[F[:,i],:]
    B /= F.shape[1]
    return B
