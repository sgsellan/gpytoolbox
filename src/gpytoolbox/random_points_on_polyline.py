import numpy as np
from .edge_indeces import edge_indeces

def random_points_on_polyline(V, n, EC=np.empty(0)):
    # Compute n uniformly distributed random points in a given polyline
    #
    # Note: The output normals follow a clockwise convention: normals will point
    # outward for a clockwise-ordered circle
    #
    # Input:
    #       V #V by 2 numpy array of polyline vertices 
    #       n integer number of desired points
    #       Optional:
    #               EC #EC by 2 numpy array of polyline indeces into V
    #
    # Output:
    #       P n by 2 numpy array of randomly sampled points
    #       N n by 2 numpy array of outward facing polyline normals at P
    #
    #

    if EC.shape[0]==0:
        EC = edge_indeces(V.shape[0],closed=False)
    
    edge_lengths = np.linalg.norm(V[EC[:,1],:] - V[EC[:,0],:],axis=1)
    normalized_edge_lengths = np.cumsum(edge_lengths)/np.sum(edge_lengths)

    # These random numbers will choose the segment
    random_numbers = np.random.rand(n)
    # These random numbers will choose where in the chosen segment
    random_numbers_in_edge = np.random.rand(n)

    P = np.zeros((n,2))
    N = np.zeros((n,2))
    for i in range(n):
        # Pick the edge
        edge_index = np.argmax((random_numbers[i]<=normalized_edge_lengths))
        # Pick the point in the edge
        P[i,:] = random_numbers_in_edge[i]*V[EC[edge_index,0],:] + (1-random_numbers_in_edge[i])*V[EC[edge_index,1],:]
        #Compute normal
        n = np.array([-(V[EC[edge_index,1],1] - V[EC[edge_index,0],1]),V[EC[edge_index,1],0] - V[EC[edge_index,0],0]])
        N[i,:] =  n/np.linalg.norm(n)
    
    return P, N