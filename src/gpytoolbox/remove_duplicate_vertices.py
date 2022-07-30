import numpy as np

def remove_duplicate_vertices(V,epsilon=0.0,faces=None):
    # Given an ordered polyline, this returns the edge indeces in a similar way
    # to how the face indeces of a triangle mesh are given.
    # Inputs:
    #       V  #V by dim numpy array of vertex positions
    #       Optional:
    #           epsilon positive float uniqueness tolerance 
    #           faces any array of indeces, for convenience
    # Outputs:
    #       SV  #SV by dim new numpy array of vertex positions
    #       SVI #SV by 1 list of indices so SV = V[SVI,:]
    #       SVJ #V by 1 list of indices so V = SV[SVJ,:]
    #       SF is the "faces" array, re-indexed to SV
    
    if epsilon==0.0:
        SV, SVI, SVJ = np.unique(V,return_index=True,return_inverse=True,axis=0)
    else:
        _, SVI, SVJ = np.unique(np.round(V/epsilon),return_index=True,return_inverse=True,axis=0)
        SV = V[SVI,:]

    if (faces is None):
        return SV, SVI, SVJ
    else:
        return SV, SVI, SVJ, SVJ[faces]