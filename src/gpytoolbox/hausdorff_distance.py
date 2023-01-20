import numpy as np
from gpytoolbox.initialize_aabbtree import initialize_aabbtree
from gpytoolbox.triangle_triangle_distance import triangle_triangle_distance

def hausdorff_distance(v1,f1,v2,f2):
    """
    Compute the minimum between two meshes.
    
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
        Hausdorff distance.

    Examples
    --------
    TODO
    """
    dim = v1.shape[1]
    # Initialize AABB tree for mesh 1
    C1,W1,CH1,tri_indices1 = initialize_aabbtree(v1,f1)
    # Initialize AABB tree for mesh 2
    C2,W2,CH2,tri_indices2 = initialize_aabbtree(v2,f2)

    first_queue_pair = [0,0]
    queue = [first_queue_pair]
    current_best_guess = np.Inf
    while len(queue)>0:
        q1, q2 = queue.pop()
        is_leaf1 = (CH1[q1,1]==-1)
        is_leaf2 = (CH2[q2,1]==-1)
        if (is_leaf1 and is_leaf2):
            # Compute distance between triangles
            t1 = tri_indices1[q1]
            t2 = tri_indices2[q2]
            d = triangle_triangle_distance(v1[f1[t1,0],:],v1[f1[t1,1],:],v1[f1[t1,2],:],v2[f2[t2,0],:],v2[f2[t2,1],:],v2[f2[t2,2],:])
            if d<current_best_guess:
                current_best_guess = d
        else:
            # Find distance between boxes
            d = np.max(np.abs(C1[q1,:] - C2[q2,:]) - (W1[q1,:] + W2[q2,:])/2)
            if d<current_best_guess:
                # Add children to queue
                if not is_leaf1:
                    queue.append([CH1[q1,0],q2])
                    queue.append([CH1[q1,1],q2])
                    queue.append([CH1[q1,2],q2])
                    queue.append([CH1[q1,3],q2])
                    if dim==3:
                        queue.append([CH1[q1,4],q2])
                        queue.append([CH1[q1,5],q2])
                        queue.append([CH1[q1,6],q2])
                        queue.append([CH1[q1,7],q2])
                if not is_leaf2:
                    queue.append([q1,CH2[q2,0]])
                    queue.append([q1,CH2[q2,1]])
                    queue.append([q1,CH2[q2,2]])
                    queue.append([q1,CH2[q2,3]])
                    if dim==3:
                        queue.append([q1,CH2[q2,4]])
                        queue.append([q1,CH2[q2,5]])
                        queue.append([q1,CH2[q2,6]])
                        queue.append([q1,CH2[q2,7]])
    return current_best_guess

