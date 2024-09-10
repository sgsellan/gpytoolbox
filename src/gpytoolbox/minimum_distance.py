import numpy as np
from gpytoolbox.initialize_aabbtree import initialize_aabbtree
from gpytoolbox.triangle_triangle_distance import triangle_triangle_distance

def minimum_distance(v1,f1,v2,f2):
    """
    Compute the minimum distance between two triangle meshes in 3D.
    
    Parameters
    ----------
    v1 : (n1,3) array
        Vertices of first mesh.
    f1 : (m1,3) array
        Faces of first mesh.
    v2 : (n2,3) array
        Vertices of second mesh.
    f2 : (m2,3) array
        Faces of second mesh.
    
    Returns
    -------
    d : float
        Minimum distance value.

    Notes
    -----
    This function could be extended with polyline and pointcloud functionality without much trouble.

    Examples
    --------
    ```python
    # meshes in v,f and u,g
    # Minimum distance value
    d = gpytoolbox.minimum_distance(v,f,u,g)
    ```
    """
        
    dim = v1.shape[1]
    # Initialize AABB tree for mesh 1
    C1,W1,CH1,PAR1,D1,tri_ind1,_ = initialize_aabbtree(v1,f1)
    # Initialize AABB tree for mesh 2
    C2,W2,CH2,PAR2,D2,tri_ind2,_ = initialize_aabbtree(v2,f2)

    first_queue_pair = [0,0]
    queue = [first_queue_pair]
    current_best_guess = np.inf
    while len(queue)>0:
        q1, q2 = queue.pop()
        # print("-----------")
        # print("Queue length : {}".format(len(queue)))
        # print("q1: ",q1)
        # print("q2: ",q2)
        # print("CH1[q1,]: ",CH1[q1,:])
        # print("CH2[q2,]: ",CH2[q2,:])
        # print("current_best_guess: ",current_best_guess)
        is_leaf1 = (CH1[q1,1]==-1)
        is_leaf2 = (CH2[q2,1]==-1)
        if (is_leaf1 and is_leaf2):
            # Compute distance between triangles
            t1 = tri_ind1[q1].item()
            t2 = tri_ind2[q2].item()
            # print("t1: ",t1)
            # print("t2: ",t2)
            d = triangle_triangle_distance(v1[f1[t1,0],:],v1[f1[t1,1],:],v1[f1[t1,2],:],v2[f2[t2,0],:],v2[f2[t2,1],:],v2[f2[t2,2],:])
            # print("d: ",d)
            if d<current_best_guess:
                current_best_guess = d
        else:
            # Find distance between boxes
            d = np.max(np.abs(C1[q1,:] - C2[q2,:]) - (W1[q1,:] + W2[q2,:])/2)
            # print("d: ",d)
            if d<current_best_guess:
                # Add children to queue
                
                if (not is_leaf1) and (is_leaf2):
                    queue.append([CH1[q1,0],q2])
                    queue.append([CH1[q1,1],q2])
                    # queue.append([CH1[q1,2],q2])
                    # queue.append([CH1[q1,3],q2])
                    # if dim==3:
                    #     queue.append([CH1[q1,4],q2])
                    #     queue.append([CH1[q1,5],q2])
                    #     queue.append([CH1[q1,6],q2])
                    #     queue.append([CH1[q1,7],q2])
                if (not is_leaf2) and (is_leaf1):
                    queue.append([q1,CH2[q2,0]])
                    queue.append([q1,CH2[q2,1]])
                    # queue.append([q1,CH2[q2,2]])
                    # queue.append([q1,CH2[q2,3]])
                    # if dim==3:
                    #     queue.append([q1,CH2[q2,4]])
                    #     queue.append([q1,CH2[q2,5]])
                    #     queue.append([q1,CH2[q2,6]])
                    #     queue.append([q1,CH2[q2,7]])
                if (not is_leaf1) and (not is_leaf2):
                    queue.append([CH1[q1,0],CH2[q2,0]])
                    queue.append([CH1[q1,1],CH2[q2,0]])
                    queue.append([CH1[q1,0],CH2[q2,1]])
                    queue.append([CH1[q1,1],CH2[q2,1]])
                    
    return current_best_guess

