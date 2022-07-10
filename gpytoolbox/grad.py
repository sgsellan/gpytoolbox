import numpy as np
from scipy.sparse import csr_matrix

from gpytoolbox.edge_indeces import edge_indeces

# Lifted from https://github.com/alecjacobson/gptoolbox/blob/master/mesh/grad.m

def grad(V,F=None):
    # Builds the finite elements gradient matrix using a piecewise linear hat functions basis.
    #
    # Input:
    #       V #V by 3 numpy array of mesh vertex positions
    #       F #F by 3 int numpy array of face/edge vertex indeces into V
    #
    # Output:
    #       G 

    # if you didn't pass an F then this is a ordered polyline
    if (F is None):
        F = edge_indeces(V.shape[0])

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
        # grad(Xijk) = (Xj-Xi) * (Vi - Vk)^R90 / 2A + (Xk-Xi) * (Vj - Vi)^R90 / 2A
        # grad(Xijk) = Xj * (Vi - Vk)^R90 / 2A + Xk * (Vj - Vi)^R90 / 2A + 
        #             -Xi * (Vi - Vk)^R90 / 2A - Xi * (Vj - Vi)^R90 / 2A
        # where Xi is the scalar value at vertex i, Vi is the 3D position of vertex
        # i, and A is the area of triangle (i,j,k). ^R90 represent a rotation of 
        # 90 degrees
        #
        # renaming indices of vertices of triangles for convenience
        i0 = F[:,0]
        i1 = F[:,1]
        i2 = F[:,2]

        # F x 3 matrices of triangle edge vectors, named after opposite vertices
        # v32 = V(i3,:) - V(i2,:);  v13 = V(i1,:) - V(i3,:); v21 = V(i2,:) - V(i1,:);
        v21 = V[i2,:] - V[i1,:]
        v02 = V[i0,:] - V[i2,:]
        v10 = V[i1,:] - V[i0,:]
    
        # area of parallelogram is twice area of triangle
        # area of parallelogram is || v1 x v2 || 
        #  n  = cross(v32,v13,2); 
        n = np.cross(v21,v02,axis=1)
        
        # This does correct l2 norm of rows, so that it contains #F list of twice
        # triangle areas
        dblA = np.linalg.norm(n,axis=1)
        u = n/np.tile(dblA[:,None],(1,3))
        
    
        # rotate each vector 90 degrees around normal
        #eperp21 = bsxfun(@times,normalizerow(cross(u,v21)),normrow(v21)./dblA);
        #eperp13 = bsxfun(@times,normalizerow(cross(u,v13)),normrow(v13)./dblA);
        eperp10 = np.cross(u,v10,axis=1)*np.tile(np.linalg.norm(v10,axis=1)[:,None]/(dblA[:,None]*np.linalg.norm(np.cross(u,v10,axis=1),axis=1)[:,None]),(1,3))
        eperp02 = np.cross(u,v02,axis=1)*np.tile(np.linalg.norm(v02,axis=1)[:,None]/(dblA[:,None]*np.linalg.norm(np.cross(u,v02,axis=1),axis=1)[:,None]) ,(1,3))

        Find = np.linspace(0,F.shape[0]-1,F.shape[0],dtype=int)


        I = np.concatenate((Find,Find,Find,Find,
        F.shape[0]+Find,F.shape[0]+Find,F.shape[0]+Find,F.shape[0]+Find,
        2*F.shape[0]+Find,2*F.shape[0]+Find,2*F.shape[0]+Find,2*F.shape[0]+Find))
        # 0*size(F,1)+repmat(1:size(F,1),1,4) ...
        # 1*size(F,1)+repmat(1:size(F,1),1,4) ...
        # 2*size(F,1)+repmat(1:size(F,1),1,4)


        J = np.concatenate((F[:,1],F[:,0],F[:,2],F[:,0]))
        J = np.concatenate((J,J,J))
        # repmat([F(:,2);F(:,1);F(:,3);F(:,1)],3,1)


        vals = np.concatenate((eperp02[:,0],-eperp02[:,0],eperp10[:,0],-eperp10[:,0],
        eperp02[:,1],-eperp02[:,1],eperp10[:,1],-eperp10[:,1],
        eperp02[:,2],-eperp02[:,2],eperp10[:,2],-eperp10[:,2]))
        # [eperp13(:,1);-eperp13(:,1);eperp21(:,1);-eperp21(:,1); ...
        # eperp13(:,2);-eperp13(:,2);eperp21(:,2);-eperp21(:,2); ...
        # eperp13(:,3);-eperp13(:,3);eperp21(:,3);-eperp21(:,3)]
    
        G = csr_matrix((vals,(I,J)),shape=(3*F.shape[0],V.shape[0]))
    
        ## Alternatively
        ##
        ## f(x) is piecewise-linear function:
        ##
        ## f(x) = ∑ φi(x) fi, f(x ∈ T) = φi(x) fi + φj(x) fj + φk(x) fk
        ## ∇f(x) = ...                 = ∇φi(x) fi + ∇φj(x) fj + ∇φk(x) fk 
        ##                             = ∇φi fi + ∇φj fj + ∇φk) fk 
        ##
        ## ∇φi = 1/hjk ((Vj-Vk)/||Vj-Vk||)^perp = 
        ##     = ||Vj-Vk|| /(2 Aijk) * ((Vj-Vk)/||Vj-Vk||)^perp 
        ##     = 1/(2 Aijk) * (Vj-Vk)^perp 
        ## 
        #m = size(F,1);
        #eperp32 = bsxfun(@times,cross(u,v32),1./dblA);
        #G = sparse( ...
        #  [0*m + repmat(1:m,1,3) ...
        #   1*m + repmat(1:m,1,3) ...
        #   2*m + repmat(1:m,1,3)]', ...
        #  repmat([F(:,1);F(:,2);F(:,3)],3,1), ...
        #  [eperp32(:,1);eperp13(:,1);eperp21(:,1); ...
        #   eperp32(:,2);eperp13(:,2);eperp21(:,2); ...
        #   eperp32(:,3);eperp13(:,3);eperp21(:,3)], ...
        #  3*m,size(V,1));
    
        if dim == 2:
            G = G[0:(2*F.shape[0]),:]



    return G
