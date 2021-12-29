import numpy as np

def regular_cube_mesh(gs):
    # Generates a regular tetrahedral mesh of a one by one by one cube. 
    # Each grid cube is decomposed into 6 reflectionally symmetric tetrahedra
    #
    # Input:
    #       gs int number of vertices on each side
    #
    # Output:
    #       V #V by 3 numpy array of mesh vertex positions
    #       T #T by 4 int numpy array of tet vertex indeces into V
    #

    # Ordering is different from matlab
    z, x, y = np.meshgrid(np.linspace(0,1,gs),np.linspace(0,1,gs),np.linspace(0,1,gs),indexing='ij')
    idx = np.reshape(np.linspace(0,gs*gs*gs-1,gs*gs*gs,dtype=int),(gs,gs,gs))
    V = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1)),np.reshape(z,(-1, 1))),axis=1)
    # Indexing is different here, careful
    v1 = np.reshape(idx[0:gs-1,0:gs-1,0:gs-1],(-1,1))
    v2 = np.reshape(idx[0:gs-1,1:gs,0:gs-1],(-1,1))
    v5 = np.reshape(idx[1:gs,0:gs-1,0:gs-1],(-1,1))
    v6 = np.reshape(idx[1:gs,1:gs,0:gs-1],(-1,1))
    v3 = np.reshape(idx[0:gs-1,0:gs-1,1:gs],(-1,1))
    v4 = np.reshape(idx[0:gs-1,1:gs,1:gs],(-1,1))
    v7 = np.reshape(idx[1:gs,0:gs-1,1:gs],(-1,1))
    v8 = np.reshape(idx[1:gs,1:gs,1:gs],(-1,1))
    t1 = np.hstack((v3,v4,v7,v1))
    t2 = np.hstack((v4,v5,v7,v1))
    t3 = np.hstack((v4,v8,v7,v5))
    t4 = np.hstack((v2,v6,v8,v5))
    t5 = np.hstack((v4,v2,v8,v5))
    t6 = np.hstack((v4,v2,v5,v1))

    T = np.vstack((t1,t2,t3,t4,t5,t6))

    return V,T
