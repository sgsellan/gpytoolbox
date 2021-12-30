import numpy as np

def regular_cube_mesh(gs,type='rotationally-symmetric'):
    # Generates a regular tetrahedral mesh of a one by one by one cube. 
    # Each grid cube is decomposed into 6 reflectionally symmetric tetrahedra
    #
    # Input:
    #       gs int number of vertices on each side
    #       Optional:
    #       type a string choosing the specific cube division scheme: 'five' for
    #            a division of each cube into 5 tets 'rotationally-symmetric'
    #            (default), 'reflectionally-symmetric' or 'hex'
    # Output:
    #       V #V by 3 numpy array of mesh vertex positions
    #       T #T by 4 int numpy array of tet vertex indeces into V
    #

    dictionary ={
    'five' : 0,
    'reflectionally-symmetric' : 1,
    'rotationally-symmetric' : 2,
    'hex' : 3
    }
    mesh_type = dictionary.get(type,-1)

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

    if mesh_type==0: # five
        t1 = np.hstack((v5,v3,v2,v1))
        t2 = np.hstack((v3,v2,v8,v5))
        t3 = np.hstack((v3,v4,v8,v2))
        t4 = np.hstack((v3,v8,v7,v5))
        t5 = np.hstack((v2,v6,v8,v5))
        T = np.vstack((t1,t2,t3,t4,t5))
    elif mesh_type==1: #reflectionally symmetric
        t1 = np.hstack((v3,v4,v7,v1))
        t2 = np.hstack((v4,v5,v7,v1))
        t3 = np.hstack((v4,v8,v7,v5))
        t4 = np.hstack((v2,v6,v8,v5))
        t5 = np.hstack((v4,v2,v8,v5))
        t6 = np.hstack((v4,v2,v5,v1))
        T = np.vstack((t1,t2,t3,t4,t5,t6))
    elif mesh_type==2: #rotationally symmetric (default)
        t1 = np.hstack((v1,v3,v7,v8))
        t2 = np.hstack((v1,v8,v7,v5))
        t3 = np.hstack((v1,v3,v8,v4))
        t4 = np.hstack((v1,v4,v8,v2))
        t5 = np.hstack((v1,v6,v8,v5))
        t6 = np.hstack((v1,v2,v8,v6))
        T = np.vstack((t1,t2,t3,t4,t5,t6))
    elif mesh_type==3: # hex mesh (polyscope's ordering convention)
        T = np.hstack((v1,v2,v3,v4,v5,v6,v7,v8))
        # to-do: check that this indeed follows polyscope's convention
    return V,T
