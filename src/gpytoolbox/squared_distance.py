import numpy as np
from gpytoolbox.initialize_aabbtree import initialize_aabbtree
from gpytoolbox.traverse_aabbtree import traverse_aabbtree
from gpytoolbox.squared_distance_to_element import squared_distance_to_element

# This class efines the functions needed to do a depth-first closest point traversal
class closest_point_traversal:
    def __init__(self,V,F,ptest):
        self.V = V
        self.F = F
        self.dim = V.shape[1]
        self.ptest = ptest
        self.current_best_guess = np.Inf
        self.current_best_element = -1
        self.others = []
    # Auxiliary function which finds the distance of point to rectangle
    def sdBox(self,p,center,width):
        q = np.abs(p - center) - width
        maxval = -np.Inf
        for i in range(self.dim):
            maxval = np.maximum(maxval,q[i])
        return np.linalg.norm((np.maximum(q,0.0))) + np.minimum(maxval,0.0)
    def traversal_function(self,q,C,W,CH,tri_indices,is_leaf):        
        # Distance is L1 norm of ptest minus center 
        if is_leaf:
            sqrD,_ = squared_distance_to_element(self.ptest,self.V,self.F[tri_indices[q],:])
        else:
            center = C[q,:]
            width = W[q,:]
            sqrD = self.sdBox(self.ptest,center,width)
            sqrD = np.sign(sqrD)*(sqrD**2.0) #Squared but signed... this isn't very legible but it is useful and efficient
        if sqrD<self.current_best_guess:
            if is_leaf:
                self.current_best_guess = sqrD
                self.current_best_element = tri_indices[q]
            else:
                self.others.append(q)
            return True
    def add_to_queue(self,queue,new_ind):
        # Depth first: insert at beginning (much less queries).
        queue.insert(0,new_ind)
                



def squared_distance(p,V,F=None,use_aabb=False):
    if (F is None):
        F = np.linspace(0,V.shape[0]-1,V.shape[0],dtype=int)[:,None]
    if use_aabb:
        C,W,CH,PAR,D,tri_ind = initialize_aabbtree(V,F=F)
        t = closest_point_traversal(V,F,p)
        traverse_fun = t.traversal_function
        add_to_queue_fun = t.add_to_queue
        _ = traverse_aabbtree(C,W,CH,tri_ind,traverse_fun,add_to_queue=add_to_queue_fun)
        ind = t.current_best_element
        min_sqrd_dist = t.current_best_guess
    else:
        # Loop over every element
        min_sqrd_dist = np.Inf
        ind = -1
        for i in range(F.shape[0]):
            this_sqrd_dist,_ = squared_distance_to_element(p,V,F[i,:])
            if this_sqrd_dist<min_sqrd_dist:
                ind = i
                min_sqrd_dist = this_sqrd_dist
    return min_sqrd_dist, ind
        
