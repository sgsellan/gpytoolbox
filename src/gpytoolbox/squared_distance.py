import numpy as np
from gpytoolbox.initialize_aabbtree import initialize_aabbtree
from gpytoolbox.traverse_aabbtree import traverse_aabbtree
from gpytoolbox.squared_distance_to_element import squared_distance_to_element
from gpytoolbox.barycentric_coordinates import barycentric_coordinates

# This defines the functions needed to do a depth-first closest point traversal
class closest_point_traversal:
    def __init__(self,V,F,ptest):
        self.V = V
        self.F = F
        self.dim = V.shape[1]
        self.ptest = ptest
        self.current_best_guess = np.inf
        self.current_best_element = -1
        self.others = []
        self.num_traversal = 0
        self.traversed = []
    # Auxiliary function which finds the distance of point to rectangle
    def sdBox(self,p,center,width):
        q = np.abs(p - center) - 0.5*width
        maxval = -np.inf
        for i in range(self.dim):
            maxval = np.maximum(maxval,q[i])
        return np.linalg.norm((np.maximum(q,0.0))) + np.minimum(maxval,0.0)
    def traversal_function(self,q,C,W,CH,tri_indices,split_dim,is_leaf):        
        self.num_traversal += 1
        # Distance is L1 norm of ptest minus center 
        if is_leaf:
            # print("Point:",self.ptest)
            # print("Verts:",self.V)
            # print("Element:",self.F[tri_indices[q],:])
            sqrD,lmb = squared_distance_to_element(self.ptest,self.V,self.F[tri_indices[q],:])
            # print("Distance",sqrD)
        else:
            center = C[q,:]
            width = W[q,:]
            sqrD = self.sdBox(self.ptest,center,width)
            sqrD = np.sign(sqrD)*(sqrD**2.0) #Squared but signed... this isn't very legible but it is useful and efficient
            self.traversed.append(q)
        if sqrD<self.current_best_guess:
            if is_leaf:
                self.current_best_guess = sqrD
                self.current_best_lmb = lmb
                # print(self.current_best_guess)
                self.current_best_element = tri_indices[q]
            else:
                # Which dimension is best?
                dist_in_best_dim = self.ptest[split_dim[q]] - center[split_dim[q]]
                self.is_left_closest = (dist_in_best_dim<0)
                self.others.append(q)
            return True
        return False
    def add_to_queue(self,queue,CH,par_ind):
        # for i in range(CH.shape[1]):
        #     queue.insert(0,CH[par_ind,i]) 
        # self.is_left_closest = False
        if self.is_left_closest:
            queue.insert(0,CH[par_ind,1])
            queue.insert(0,CH[par_ind,0])
        else:
            queue.insert(0,CH[par_ind,0])
            queue.insert(0,CH[par_ind,1])
        # if self.is_left_closest:
        #     queue.append(CH[par_ind,0])
        #     queue.append(CH[par_ind,1])
        # else:
        #     queue.append(CH[par_ind,1])
        #     queue.append(CH[par_ind,0])
        
        # Depth first: insert at beginning (much less queries).
        # queue.insert(0,new_ind)
                



def squared_distance(P,V,F=None,use_cpp=False,use_aabb=False,C=None,W=None,CH=None,tri_ind=None,split_dir=None):
    """Squared distances from a set of points in space.

    General-purpose function which computes the squared distance from a set of points to a mesh, point cloud or polyline, in two or three dimensions. Optionally, uses an aabb tree for efficient computation.

    Parameters
    ----------
    P : (p,dim) numpy double array
        Matrix of query point positions
    V : (v,dim) numpy double array
        Matrix of mesh/polyline/pointcloud coordinates
    F : (f,s) numpy int array (optional, default None)
        Matrix of mesh/polyline/pointcloud indices into V. If None, input is assumed to be point cloud.
    use_cpp : bool, optional (default False)
        Whether to use the C++ libigl implementation of the AABB tree (much faster). If True, the following parameters are ignored.
    use_aabb : bool, optional (default False)
        Whether to use an AABB tree for logarithmic computation 
    C : numpy double array, optional (default None)
        Matrix of AABB box centers (if None, will be computed)
    W : numpy double array, optional (default None)
        Matrix of AABB box widths (if None, will be computed)
    CH : numpy int array, optional (default None)
        Matrix of child indeces (-1 if leaf node). If None, will be computed
    tri_ind : numpy int array, optional (default None)
        Vector of AABB element indices (-1 if *not* leaf node). If None, will be computed
    split_dir : numpy double array, optional (default None)
        Vector of AABB split directions (if None, will be computed)

    Returns
    -------
    squared_distances : (p,) numpy double array
        Vector of minimum squared distances
    indices : (p,) numpy int array
        Indices into F (or V, if F is None) of closest elements to each query point
    lmbs : (p,s) numpy double array
        Barycentric coordinates into the closest element of each closest mesh point to each query point
    

    See Also
    --------
    squared_distance_to_element, initialize_aabb.

    Examples
    --------
    ```python
    v,f = gpytoolbox.read_mesh("bunny.obj") # Read a mesh
    v = gpytoolbox.normalize_points(v) # Normalize mesh
    # Generate query points
    P = 2*np.random.rand(num_samples,3)-4
    # Compute distances
    sqrD_gt,ind,b = gpytoolbox.squared_distance(P,v,F=f,use_aabb=True)
    ```
    """
    if (F is None):
        F = np.linspace(0,V.shape[0]-1,V.shape[0],dtype=int)[:,None]
    dim = V.shape[1]
    P = np.reshape(P,(-1,dim),order='F')


    if use_cpp:
        try:
            from gpytoolbox_bindings import _point_mesh_squared_distance_cpp_impl
        except:
            raise ImportError("Gpytoolbox cannot import its C++ point_mesh_squared_distance binding.")
        squared_distances, indices, closest_points = _point_mesh_squared_distance_cpp_impl(V.astype(np.float64),F.astype(np.int32),P.astype(np.float64))
        lmbs = np.zeros((P.shape[0],F.shape[1]))
        if (F.shape[1] == 1):
            lmbs = np.ones((P.shape[0],1))
        elif(F.shape[1] == 2):
            lmbs = np.zeros((P.shape[0],2))
            lmbs[:,0] = np.linalg.norm(V[F[indices,0],:]-closest_points,axis=1)
            lmbs[:,1] = np.linalg.norm(V[F[indices,1],:]-closest_points,axis=1)
            lmbs = lmbs/np.sum(lmbs,axis=1)[:,None]
            lmbs = np.fliplr(lmbs)
        elif(F.shape[1] == 3):
            lmbs = barycentric_coordinates(closest_points,V[F[indices,0],:],V[F[indices,1],:],V[F[indices,2],:])
            # for j in range(P.shape[0]):
            #     lmbs[j,:] = barycentric_coordinates(closest_points[j,:],V[F[indices[j],0],:],V[F[indices[j],1],:],V[F[indices[j],2],:])
        return squared_distances, indices, lmbs

    
    squared_distances = -np.ones(P.shape[0])
    indices = -np.ones(P.shape[0],dtype=int)
    lmbs = np.zeros((P.shape[0],F.shape[1]))
    if use_aabb:
        # Build tree once
        if ((C is None) or (W is None) or (tri_ind is None) or (CH is None) or (split_dir is None)):
            C,W,CH,_,_,tri_ind,split_dir = initialize_aabbtree(V,F=F)
            # V,Q,H = bad_quad_mesh_from_quadtree(C,W,CH)
            # ps.init()
            # ps.register_surface_mesh("test",V,Q)
            # ps.show()
        for j in range(P.shape[0]):
            t = closest_point_traversal(V,F,P[j,:])
            traverse_fun = t.traversal_function
            add_to_queue_fun = t.add_to_queue
            # print(C)
            # print(W)
            # print(CH)
            # print(tri_ind)
            _ = traverse_aabbtree(C,W,CH,tri_ind,split_dir,traverse_fun,add_to_queue=add_to_queue_fun)
            # print(t.num_traversal)
            indices[j] = np.squeeze(t.current_best_element)
            squared_distances[j] = np.squeeze(t.current_best_guess)
            lmbs[j,:] = t.current_best_lmb
    else:
        # Loop over every element
        t = None
        for j in range(P.shape[0]):
            min_sqrd_dist = np.inf
            ind = -1
            best_lmb = []
            for i in range(F.shape[0]):
                this_sqrd_dist,lmb = squared_distance_to_element(P[j,:],V,F[i,:])
                if this_sqrd_dist<min_sqrd_dist:
                    ind = i
                    min_sqrd_dist = this_sqrd_dist
                    best_lmb = lmb
            squared_distances[j] = min_sqrd_dist
            indices[j] = ind
            lmbs[j,:] = best_lmb
    return squared_distances, indices, lmbs
        
