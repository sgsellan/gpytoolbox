import numpy as np
from gpytoolbox.initialize_aabbtree import initialize_aabbtree
from gpytoolbox.squared_distance_to_element import squared_distance_to_element

def approximate_hausdorff_distance(v1,f1,v2,f2,use_cpp=True):
    """
    Approximate the Hausdorff distance between two triangle meshes in 3D, i.e.
    the maximum distance between any point on one mesh and the closest point
    on the other mesh.

    d = max { d(pA,B), d(pB,A) }
    
    where pA is a point on mesh A and pB is a point on mesh B, and d(pA,B) is
    the distance between pA and the closest point on B. Our approximation will instead compute

    d = max { d(vA,B), d(vB,A) }

    where vA is a vertex on mesh A and vB is a vertex on mesh B.

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
    use_cpp : bool, optional (default: True)
        Whether to use the C++ implementation of triangle_triangle_distance.
    
    Returns
    -------
    d : float
        Minimum distance value.

    Notes
    -----
    If you want an **exact** Hausdorff distance, you can heavily upsample both meshes using gpytoolbox.subdivide(method='upsample'). The approximated Hausdorff distance returned by this function will converge to the exact Hausdorff distance as the number of iterations of subdivision goes to infinity.

    Examples
    --------
    ```python
    # meshes in v,f and u,g
    # Minimum distance value
    d = gpytoolbox.minimum_distance(v,f,u,g)
    ```
    """

    # cpp implementation
    if use_cpp:
        try:
            from gpytoolbox_bindings import _hausdorff_distance_cpp_impl
        except:
            raise ImportError("Gpytoolbox cannot import its C++ fast winding number binding.") 
        return _hausdorff_distance_cpp_impl(v1,f1,v2,f2)

    # Let's start by computing the one-sided distance, i.e., max(d(vA,B)). We will do this with a loop, but with pre-computed trees for efficient queriying.
        
    dim = v1.shape[1]
    # Initialize AABB tree for mesh 1
    C1,W1,CH1,PAR1,D1,tri_ind1,_ = initialize_aabbtree(v1,f1)
    # Initialize AABB tree for mesh 2
    C2,W2,CH2,PAR2,D2,tri_ind2,_ = initialize_aabbtree(v2,f2)

    # print("Computing one-sided distance...")

    current_best_guess_hd = 0.0
    for i in range(v1.shape[0]):
        # print("Vertex %d of %d" % (i+1,v1.shape[0]))
        current_best_guess_dviB = np.Inf # current best guess for d(vi,B)

        queue = [0]
        while (len(queue)>0 and current_best_guess_dviB>current_best_guess_hd):
            q2 = queue.pop()
            # print("-----------")
            # print("Queue length : {}".format(len(queue)))
            # print("q1: ",q1)
            # print("q2: ",q2)
            # print("CH1[q1,]: ",CH1[q1,:])
            # print("CH2[q2,]: ",CH2[q2,:])
            # print("current_best_guess: ",current_best_guess)
            is_leaf2 = (CH2[q2,1]==-1)
            if (is_leaf2):
                # Compute distance between vi and triangle q2
                d = np.sqrt(squared_distance_to_element(v1[i,:],v2,f2[tri_ind2[q2],:])[0])
                if d < current_best_guess_dviB:
                    current_best_guess_dviB = d
            else:
                # Compute distance between vi and bounding box of q2
                # Distance from vi to the bounding box of q2
                d = point_to_box_distance(v1[i,:],C2[q2,:],W2[q2,:])
                d_max = point_to_box_max_distance(v1[i,:],C2[q2,:],W2[q2,:])
                # print(d_max)
                # If d_max is smaller than the current best guess for HD, then we don't need to pursue this part of the tree
                if ((d < current_best_guess_dviB) and (d_max > current_best_guess_hd)):
                    # Add children to queue
                    queue.append(CH2[q2,0])
                    queue.append(CH2[q2,1])
        # We have computed d(vi,B). Does it change our guess for hausdorff?
        if current_best_guess_dviB > current_best_guess_hd:
            current_best_guess_hd = current_best_guess_dviB

    # print("One-sided distance: {}".format(current_best_guess_hd))
    # print("Computing two-sided distance...")

    # Now we do the other side, i.e., max(d(vB,A))
    for i in range(v2.shape[0]):
        # print("Vertex %d of %d" % (i+1,v2.shape[0]))
        current_best_guess_dviA = np.Inf
        queue = [0]
        while (len(queue)>0 and current_best_guess_dviA>current_best_guess_hd):
            q2 = queue.pop()
            is_leaf2 = (CH1[q2,1]==-1)
            if (is_leaf2):
                # Compute distance between vi and triangle q2
                d = np.sqrt(squared_distance_to_element(v2[i,:],v1,f1[tri_ind1[q2],:])[0])
                if d < current_best_guess_dviA:
                    current_best_guess_dviA = d
            else:
                # Compute distance between vi and bounding box of q2
                # Distance from vi to the bounding box of q2
                d = point_to_box_distance(v2[i,:],C1[q2,:],W1[q2,:])
                d_max = point_to_box_max_distance(v2[i,:],C1[q2,:],W1[q2,:])
                # If d_max is smaller than the current best guess for HD, then we don't need to pursue this part of the tree
                if (d < current_best_guess_dviA and d_max > current_best_guess_hd):
                    # Add children to queue
                    queue.append(CH1[q2,0])
                    queue.append(CH1[q2,1])
        # We have computed d(vi,A). Does it change our guess for hausdorff?
        if current_best_guess_dviA > current_best_guess_hd:
            current_best_guess_hd = current_best_guess_dviA
    
    # We are done!
    return current_best_guess_hd

def point_to_box_max_distance(p,C,W):
    """
    Compute the maximum distance between a point and a box.

    Parameters
    ----------
    p : (3,) array
        Point.
    C : (3,) array
        Center of box.
    W : (3,) array
        Width of box.

    Returns
    -------
    d : float
        Maximum distance between point and box.
    """
    d = 0.0
    for i in range(3):
        if p[i] < C[i]:
            # Distance to oppoisite corner
            d += (p[i]-C[i]-0.5*W[i])**2
        elif p[i] > C[i]:
            # Distance to oppoisite corner
            d += (p[i]-C[i]+0.5*W[i])**2
    return np.sqrt(d)


def point_to_box_distance(p,C,W):
    """
    Compute the distance between a point and a box.

    Parameters
    ----------
    p : (3,) array
        Point.
    C : (3,) array
        Center of box.
    W : (3,) array
        Width of box.

    Returns
    -------
    d : float
        Distance between point and box.
    """
    d = 0.0
    for i in range(3):
        if p[i] < C[i]-0.5*W[i]:
            d += (p[i]-C[i]+0.5*W[i])**2
        elif p[i] > C[i]+0.5*W[i]:
            d += (p[i]-C[i]-0.5*W[i])**2
    return np.sqrt(d)
