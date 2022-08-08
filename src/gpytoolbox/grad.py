import numpy as np
from scipy.sparse import csr_matrix

from gpytoolbox.edge_indices import edge_indices

# Lifted from https://github.com/alecjacobson/gptoolbox/blob/master/mesh/grad.m

def grad(V,F=None):
    """Finite element gradient matrix

    Given a triangle mesh or a polyline, computes the finite element gradient matrix assuming piecewise linear hat function basis.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a polyline or triangle mesh
    F : numpy int array, optional (default: None)
        if None, interpret input as ordered polyline;
        if (m,3) numpy int array, interpred as face index list of a triangle
        mesh

    Returns
    -------
    G : (d*m,n) scipy sparse.csr_matrix
        Sparse FEM gradient matrix

    See Also
    --------
    cotangent_laplacian.

    Notes
    -----

    Examples
    --------
    TO-DO
    """
    # Builds the finite elements gradient matrix using a piecewise linear hat functions basis.
    #
    # Input:
    #       V #V by 3 numpy array of mesh vertex positions
    #       F #F by 3 int numpy array of face/edge vertex indeces into V
    #
    # Output:
    #       G #F*dim by #V sparse gradient matrix

    # if you didn't pass an F then this is a ordered polyline
    if (F is None):
        F = edge_indices(V.shape[0])

    dim = V.shape[1]
    simplex_size = F.shape[1]
    # Option 1: simplex size is two
    if simplex_size==2:
        # Then this is just finite difference with varying edge lengths
        edge_lengths = np.linalg.norm(V[F[:,1],:] - V[F[:,0],:],axis=1)
        I = np.linspace(0,F.shape[0]-1,F.shape[0],dtype=int)
        I = np.concatenate((I,I))
        J = np.concatenate((F[:,0],F[:,1]))
        vals = np.ones(F.shape[0])/edge_lengths
        vals = np.concatenate((-vals,vals))
        G = csr_matrix((vals,(I,J)),shape=(F.shape[0],V.shape[0]))

    if simplex_size==3:
        # There are two options: dimension two or three. If it's two, we'll add a third zero dimension for convenience
        if dim==2:
            V = np.hstack((V,np.zeros((V.shape[0],1))))
        # Gradient of a scalar function defined on piecewise linear elements (mesh)
        # is constant on each triangle i,j,k:
        #
        # renaming indices of vertices of triangles for convenience
        i0 = F[:,0]
        i1 = F[:,1]
        i2 = F[:,2]

        # F x 3 matrices of triangle edge vectors, named after opposite vertices
        v21 = V[i2,:] - V[i1,:]
        v02 = V[i0,:] - V[i2,:]
        v10 = V[i1,:] - V[i0,:]
    
        # area of parallelogram is twice area of triangle
        n = np.cross(v21,v02,axis=1)
        
        # This does correct l2 norm of rows, so that it contains #F list of twice
        # triangle areas
        dblA = np.linalg.norm(n,axis=1)
        u = n/np.tile(dblA[:,None],(1,3))
        
        eperp10 = np.cross(u,v10,axis=1)*np.tile(np.linalg.norm(v10,axis=1)[:,None]/(dblA[:,None]*np.linalg.norm(np.cross(u,v10,axis=1),axis=1)[:,None]),(1,3))
        eperp02 = np.cross(u,v02,axis=1)*np.tile(np.linalg.norm(v02,axis=1)[:,None]/(dblA[:,None]*np.linalg.norm(np.cross(u,v02,axis=1),axis=1)[:,None]) ,(1,3))

        Find = np.linspace(0,F.shape[0]-1,F.shape[0],dtype=int)


        I = np.concatenate((Find,Find,Find,Find,
        F.shape[0]+Find,F.shape[0]+Find,F.shape[0]+Find,F.shape[0]+Find,
        2*F.shape[0]+Find,2*F.shape[0]+Find,2*F.shape[0]+Find,2*F.shape[0]+Find))


        J = np.concatenate((F[:,1],F[:,0],F[:,2],F[:,0]))
        J = np.concatenate((J,J,J))


        vals = np.concatenate((eperp02[:,0],-eperp02[:,0],eperp10[:,0],-eperp10[:,0],
        eperp02[:,1],-eperp02[:,1],eperp10[:,1],-eperp10[:,1],
        eperp02[:,2],-eperp02[:,2],eperp10[:,2],-eperp10[:,2]))
    
        G = csr_matrix((vals,(I,J)),shape=(3*F.shape[0],V.shape[0]))
    
        if dim == 2:
            G = G[0:(2*F.shape[0]),:]



    return G
