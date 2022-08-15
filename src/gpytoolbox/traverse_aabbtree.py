import numpy as np

def traverse_aabbtree(C,W,CH,tri_indices,traversal_bool_fun,add_to_queue=None):
    """Axis-Aligned Bounding-Box hierarchy traversal.

    Simple function which traverses an AABB tree given rejection and queue addition strategies. 

    Parameters
    ----------
    C : (c,dim) numpy double array
        Matrix of cell centers
    W : (c,dim) numpy double array
        Matrix of half cell widths
    CH : (c,2) numpy int array
        Matrix of child indices (-1 if leaf node)
    tri_indices : numpy int array
        Vector of element indices (-1 if *not* leaf node)
    traversal_bool_fun : func
        Function which takes (box_index,C,W,CH,is_leaf) as input and returns whether to continue traversing to the next depth (True) or reject branch entirely (False)
    add_to_queue : func, optional (default None)
        Function which takes (queue,box_index) as input and adds box_index to the queue (to support different search strategies). By default, appends to the end (breadth first).

    Returns
    -------
    success : bool
        True if function finishes running without errors.

    See Also
    --------
    initialize_aabbtree.


    Examples
    --------
    ```python
    # This is a sample class that defines the functions needed to do a depth-first closest point traversal
    class test_closest_point_traversal:
        def __init__(self,P,ptest):
            self.P = P
            self.ptest = ptest
            self.current_best_guess = np.Inf
            self.current_best_element = -1
            self.others = []
        # Auxiliary function which finds the distance of point to rectangle
        def rectangle_sdf(self,pt,center,width):
            shift_pt = np.abs(pt - center)
            pt_dist = shift_pt - width/2
            out = 0
            out =  out + np.max(pt_dist) * (np.max(pt_dist) <= 0)
            out =  out + pt_dist[1] * np.logical_and(pt_dist[0] <= 0, pt_dist[1] > 0)
            out =  out + pt_dist[0] * np.logical_and(pt_dist[1] <= 0, pt_dist[0] > 0)
            out =  out + np.sqrt(np.sum(pt_dist**2)) * (np.min(pt_dist) > 0)
            return out
        def traversal_function(self,q,C,W,CH,is_leaf):
            center = C[q,:]
            width = W[q,:]
            # Distance is L1 norm of ptest minus center 
            sdf = self.rectangle_sdf(self.ptest,center,width)
            # print(sdf)
            if sdf<self.current_best_guess:
                if is_leaf:
                    self.current_best_guess = sdf
                    self.current_best_box = q
                    # print(self.current_best_guess)
                    # print(self.current_best_box)
                    # print(C[self.current_best_box,:])
                    # print(W[self.current_best_box,:])
                else:
                    self.others.append(q)
                return True
        def add_to_queue(self,queue,new_ind):
            # Depth first: insert at beginning.
            queue.insert(0,new_ind)
    
    # This class allows us to define functions for our tree traversal.
    # First, build tree:
    P = np.random.rand(11,2)
    C,W,CH,PAR,D,tri_ind = gpytoolbox.initialize_aabbtree(P)
    # Next choose query point
    ptest = P[9,:] + 1e-5
    # Initialize traversal class
    t = test_closest_point_traversal(P,ptest)
    traverse_fun = t.traversal_function
    add_to_queue_fun = t.add_to_queue
    # Call to traversal function
    _ = gpytoolbox.traverse_aabbtree(C,W,CH,traverse_fun,add_to_queue=add_to_queue_fun)
    i = t.current_best_box
    # tri_ind[i] should be 9 by construction 
    ```
    """
    if (add_to_queue is None):
        # By default breadth first
        def add_to_queue(curr_queue,new_ind):
            curr_queue.append(new_ind)


    queue = [0]
    
    while len(queue)>0:
        # Pop from queue
        q = queue.pop(0)
        is_leaf = (CH[q,1]==-1)
        # Check if, e.g., point is inside this cell
        if traversal_bool_fun(q,C,W,CH,tri_indices,is_leaf):
            # Is it leaf?
            if (not is_leaf):
                # If not, add children to queue
                for i in range(CH.shape[1]):
                    add_to_queue(queue,CH[q,i])
                    # queue.append(CH[q,i])
    return True
            