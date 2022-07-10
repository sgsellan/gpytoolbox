import numpy as np
from scipy.sparse import csr_matrix

from gpytoolbox.edge_indeces import edge_indeces
from gpytoolbox.doublearea import doublearea

# Lifted from https://github.com/alecjacobson/gptoolbox/blob/master/mesh/massmatrix_intrinsic.m

def massmatrix(V,F=None,type='barycentric'):
    # Builds the finite elements gradient matrix using a piecewise linear hat functions basis.
    #
    # Input:
    #       V #V by 3 numpy array of mesh vertex positions
    #       F #F by 3 int numpy array of face/edge vertex indeces into V
    #       Optional:
    #           type either of 'voronoi' {default} or 'barycentric'
    #
    # Output:
    #       M #V by #V diagonal matrix of barycentric vertex areas

    dictionary ={
    'voronoi' : 0,
    'barycentric' : 1
    }
    massmatrix_type = dictionary.get(type,-1)

    # if you didn't pass an F then this is a ordered polyline
    if (F is None):
        F = edge_indeces(V.shape[0])

    simplex_size = F.shape[1]
    # Option 1: simplex size is two
    if simplex_size==2:
        # Then this is just finite difference with varying edge lengths
        edge_lengths = np.linalg.norm(V[F[:,1],:] - V[F[:,0],:],axis=1)
        vals = np.concatenate((edge_lengths,edge_lengths))/2.
        I = np.concatenate((F[:,0],F[:,1]))
        

    if simplex_size==3:
        dblA = doublearea(V,F)
        if massmatrix_type==1:
            vals = np.concatenate((dblA,dblA,dblA))/6.
            I = np.concatenate((F[:,0],F[:,1],F[:,2]))
        else:
            i0 = F[:,0]
            i1 = F[:,1]
            i2 = F[:,2]
            l21 = np.linalg.norm(V[i2,:] - V[i1,:],axis=1)[:,None]
            l02 = np.linalg.norm(V[i0,:] - V[i2,:],axis=1)[:,None]
            l10 = np.linalg.norm(V[i1,:] - V[i0,:],axis=1)[:,None]
            l = np.hstack(( l21, l02, l10 ))
            print((l[:,2]**2.+l[:,1]**2.-l[:,0]**2.0))
            print((2*l[:,1]*l[:,2]))
            cosines = np.hstack(( 
            ((l[:,2]**2.+l[:,1]**2.-l[:,0]**2.0)/(2*l[:,1]*l[:,2]))[:,None] ,
            ((l[:,0]**2.+l[:,2]**2.-l[:,1]**2.)/(2.*l[:,0]*l[:,2]))[:,None],
            ((l[:,0]**2.+l[:,1]**2.-l[:,2]**2.)/(2*l[:,0]*l[:,1]))[:,None]
            ))
            #print(cosines)
            # cosines = [ ...
            # (l(:,3).^2+l(:,2).^2-l(:,1).^2)./(2*l(:,2).*l(:,3)), ...
            #     (l(:,1).^2+l(:,3).^2-l(:,2).^2)./(2*l(:,1).*l(:,3)), ...
            #     (l(:,1).^2+l(:,2).^2-l(:,3).^2)./(2*l(:,1).*l(:,2))];
            barycentric = cosines*l
            normalized_barycentric = barycentric/np.hstack(( np.sum(barycentric,axis=1)[:,None], np.sum(barycentric,axis=1)[:,None], np.sum(barycentric,axis=1)[:,None] ))


            # barycentric = cosines.*l;
            # normalized_barycentric = barycentric./ ...
            #     [sum(barycentric')' sum(barycentric')' sum(barycentric')'];

            areas = 0.25*np.sqrt(  
                (l[:,0] + l[:,1] - l[:,2])*
                (l[:,0] - l[:,1] + l[:,2])*
                (-l[:,0] + l[:,1] + l[:,2])*
                (l[:,0] + l[:,1] + l[:,2])  )

            partial_triangle_areas = normalized_barycentric * np.hstack((areas[:,None],areas[:,None],areas[:,None] ))

            # areas = 0.25*sqrt( ...
            #     (l(:,1) + l(:,2) - l(:,3)).* ...
            #     (l(:,1) - l(:,2) + l(:,3)).* ...
            #     (-l(:,1) + l(:,2) + l(:,3)).* ...
            #     (l(:,1) + l(:,2) + l(:,3)));
            # partial_triangle_areas = normalized_barycentric.*[areas areas areas];

            quads = np.hstack(( 
            ((partial_triangle_areas[:,1]+ partial_triangle_areas[:,2])*0.5)[:,None],
            ((partial_triangle_areas[:,0]+ partial_triangle_areas[:,2])*0.5)[:,None],
            ((partial_triangle_areas[:,0]+ partial_triangle_areas[:,1])*0.5)[:,None] ))

            # quads = [ (partial_triangle_areas(:,2)+ partial_triangle_areas(:,3))*0.5 ...
            #     (partial_triangle_areas(:,1)+ partial_triangle_areas(:,3))*0.5 ...
            #     (partial_triangle_areas(:,1)+ partial_triangle_areas(:,2))*0.5];
            
            # quads(cosines(:,1)<0,:) = [areas(cosines(:,1)<0,:)*0.5, ...
            #     areas(cosines(:,1)<0,:)*0.25, areas(cosines(:,1)<0,:)*0.25];
            # quads(cosines(:,2)<0,:) = [areas(cosines(:,2)<0,:)*0.25, ...
            #     areas(cosines(:,2)<0,:)*0.5, areas(cosines(:,2)<0,:)*0.25];
            # quads(cosines(:,3)<0,:) = [areas(cosines(:,3)<0,:)*0.25, ...
            #     areas(cosines(:,3)<0,:)*0.25, areas(cosines(:,3)<0,:)*0.5];

            I = np.concatenate((i0,i1,i2))
            vals = np.reshape(quads,(-1,1),order='F').squeeze()

            # i = [i1 i2 i3];
            # j = [i1 i2 i3];
            # v = reshape(quads,size(quads,1)*3,1);
    M = csr_matrix((vals,(I,I)),shape=(V.shape[0],V.shape[0]))

    return M