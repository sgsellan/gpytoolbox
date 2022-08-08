import numpy as np
from scipy.sparse import csr_matrix

from gpytoolbox.doublearea_intrinsic import doublearea_intrinsic

# Lifted from https://github.com/alecjacobson/gptoolbox/blob/master/mesh/massmatrix_intrinsic.m

def massmatrix_intrinsic(l_sq,F,n=None,type='voronoi'):
    """FEM intrinsic mass matrix
    
    Builds the finite elements mass matrix for a triangle mesh using a piecewise
    linear hat function basis, using only intrinsic information (squared
    halfedge edge lengths).

    Parameters
    ----------
    l_sq : (m,3) numpy double array
        Vector of squared halfedge lengths as computed by halfedge_lengths_squared
    F : (m,3) numpy int array
        face index list of a triangle mesh (into a V assumed to exist)
    n : int, optional (default: None)
        Integer denoting the number of vertices in the mesh
    type : str, optional (default: 'voronoi')
        Type of mass matrix computation: 'voronoi' (default), 'full' or 'barycentric'

    Returns
    -------
    M : (n,n) scipy sparse.csr_matrix
        Intrinsicly computed mass matrix

    See Also
    --------
    massmatrix.

    Notes
    -----
    This implementation is lifted from https://github.com/alecjacobson/gptoolbox/blob/master/mesh/massmatrix_intrinsic.m

    Examples
    --------
    TO-DO
    """

    assert F.shape == l_sq.shape
    assert F.shape[1]==3
    assert np.all(l_sq >= 0)

    dictionary ={
    'voronoi' : 0,
    'barycentric' : 1,
    'full' : 2
    }
    massmatrix_type = dictionary.get(type,-1)

    if n==None:
        n = np.max(F)+1

    
    dblA = doublearea_intrinsic(l_sq,F)
    if massmatrix_type==0:
        #Voronoi

        l = np.sqrt(l_sq)
        cosines = np.stack(( 
        ((l_sq[:,2]+l_sq[:,1]-l_sq[:,0])/(2.*l[:,2]*l[:,1])),
        ((l_sq[:,0]+l_sq[:,2]-l_sq[:,1])/(2.*l[:,0]*l[:,2])),
        ((l_sq[:,1]+l_sq[:,0]-l_sq[:,2])/(2.*l[:,1]*l[:,0]))
        ), axis=-1)

        # cosines = [ ...
        # (l(:,3).^2+l(:,2).^2-l(:,1).^2)./(2*l(:,2).*l(:,3)), ...
        #     (l(:,1).^2+l(:,3).^2-l(:,2).^2)./(2*l(:,1).*l(:,3)), ...
        #     (l(:,1).^2+l(:,2).^2-l(:,3).^2)./(2*l(:,1).*l(:,2))];

        barycentric = cosines*l
        normalized_barycentric = barycentric/np.hstack(( np.sum(barycentric,axis=1)[:,None], np.sum(barycentric,axis=1)[:,None], np.sum(barycentric,axis=1)[:,None] ))

        # barycentric = cosines.*l;
        # normalized_barycentric = barycentric./ ...
        #     [sum(barycentric')' sum(barycentric')' sum(barycentric')'];

        partial_triangle_areas = normalized_barycentric * 0.5 * np.stack((dblA,dblA,dblA), axis=-1)

        # partial_triangle_areas = normalized_barycentric.*[areas areas areas];

        quads = np.stack(( 
        ((partial_triangle_areas[:,1]+ partial_triangle_areas[:,2])*0.5),
        ((partial_triangle_areas[:,0]+ partial_triangle_areas[:,2])*0.5),
        ((partial_triangle_areas[:,0]+ partial_triangle_areas[:,1])*0.5)
        ), axis=-1)

        # quads = [ (partial_triangle_areas(:,2)+ partial_triangle_areas(:,3))*0.5 ...
        #     (partial_triangle_areas(:,1)+ partial_triangle_areas(:,3))*0.5 ...
        #     (partial_triangle_areas(:,1)+ partial_triangle_areas(:,2))*0.5];

        c0s = cosines[:,0]<0
        quads[c0s,:] = np.stack((
            0.25*dblA[c0s], 0.125*dblA[c0s], 0.125*dblA[c0s]
            ), axis=-1)
        c1s = cosines[:,1]<0
        quads[c1s,:] = np.stack((
            0.125*dblA[c1s], 0.25*dblA[c1s], 0.125*dblA[c1s]
            ), axis=-1)
        c2s = cosines[:,2]<0
        quads[c2s,:] = np.stack((
            0.125*dblA[c2s], 0.125*dblA[c2s], 0.25*dblA[c2s]
            ), axis=-1)
        
        # quads(cosines(:,1)<0,:) = [areas(cosines(:,1)<0,:)*0.5, ...
        #     areas(cosines(:,1)<0,:)*0.25, areas(cosines(:,1)<0,:)*0.25];
        # quads(cosines(:,2)<0,:) = [areas(cosines(:,2)<0,:)*0.25, ...
        #     areas(cosines(:,2)<0,:)*0.5, areas(cosines(:,2)<0,:)*0.25];
        # quads(cosines(:,3)<0,:) = [areas(cosines(:,3)<0,:)*0.25, ...
        #     areas(cosines(:,3)<0,:)*0.25, areas(cosines(:,3)<0,:)*0.5];

        I = np.concatenate((F[:,0],F[:,1],F[:,2]))
        J = I
        vals = np.reshape(quads,(-1,1),order='F').squeeze()

        # i = [i1 i2 i3];
        # j = [i1 i2 i3];
        # v = reshape(quads,size(quads,1)*3,1);

    elif massmatrix_type==1:
        #Barycentric

        I = np.concatenate((F[:,0],F[:,1],F[:,2]))
        J = I
        vals = np.concatenate((dblA,dblA,dblA))/6.

    elif massmatrix_type==2:
        #Full
        I = np.concatenate((F[:,0], F[:,1], F[:,1], F[:,2], F[:,2], F[:,0],
            F[:,0], F[:,1], F[:,2]))
        J = np.concatenate((F[:,1], F[:,0], F[:,2], F[:,1], F[:,0], F[:,2],
            F[:,0], F[:,1], F[:,2]))
        offd = dblA / 24.
        diag = dblA / 12.
        vals = np.concatenate((offd, offd, offd, offd, offd, offd,
            diag, diag, diag))

        # i = [i1 i2 i2 i3 i3 i1  i1 i2 i3];
        # j = [i2 i1 i3 i2 i1 i3  i1 i2 i3];
        # offd_v = dblA/24.;
        # diag_v = dblA/12.;
        # v = [offd_v,offd_v, offd_v,offd_v, offd_v,offd_v, diag_v,diag_v,diag_v];  

    else:
        assert False, "invalid massmatrix type"

    M = csr_matrix((vals,(I,J)),shape=(n,n))

    return M