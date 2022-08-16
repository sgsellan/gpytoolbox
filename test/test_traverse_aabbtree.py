from .context import gpytoolbox
from .context import unittest
from .context import numpy as np




# This is a sample class that defines the functions needed to do a depth-first closest point traversal
class test_closest_point_traversal:
    def __init__(self,P,ptest):
        self.P = P
        self.dim = P.shape[1]
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
        center = C[q,:]
        width = W[q,:]
        # Distance is L1 norm of ptest minus center 
        sdf = self.sdBox(self.ptest,center,width)
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
                



class TestTraverseAabbTree(unittest.TestCase):
    def test_find_closest_point_2d(self):
        np.random.seed(0)
        for ss in range(10,2000,100):       
            P = np.random.rand(ss,2)
            ptest = P[9,:] + 1e-5
            C,W,CH,PAR,D,tri_ind = gpytoolbox.initialize_aabbtree(P)
            t = test_closest_point_traversal(P,ptest)
            traverse_fun = t.traversal_function
            add_to_queue_fun = t.add_to_queue
            _ = gpytoolbox.traverse_aabbtree(C,W,CH,tri_ind,traverse_fun,add_to_queue=add_to_queue_fun)
            i = t.current_best_box
            self.assertTrue(tri_ind[i]==9)

        
        




if __name__ == '__main__':
    unittest.main()
