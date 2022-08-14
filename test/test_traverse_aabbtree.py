from .context import gpytoolbox
from .context import unittest
from .context import numpy as np




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
                



class TestTraverseAabbTree(unittest.TestCase):
    def test_find_closest_point(self):
        np.random.seed(0)
        for ss in range(10,20000,100):       
            P = np.random.rand(11,2)
            ptest = P[9,:] + 1e-5
            C,W,CH,PAR,D,tri_ind = gpytoolbox.initialize_aabbtree(P)
            t = test_closest_point_traversal(P,ptest)
            traverse_fun = t.traversal_function
            add_to_queue_fun = t.add_to_queue
            _ = gpytoolbox.traverse_aabbtree(C,W,CH,traverse_fun,add_to_queue=add_to_queue_fun)
            i = t.current_best_box
            self.assertTrue(tri_ind[i]==9)

        
        




if __name__ == '__main__':
    unittest.main()
