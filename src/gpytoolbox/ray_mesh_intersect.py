import numpy as np
from gpytoolbox.ray_box_intersect import ray_box_intersect
from gpytoolbox.ray_triangle_intersect import ray_triangle_intersect
from gpytoolbox.barycentric_coordinates import barycentric_coordinates
from gpytoolbox.initialize_aabbtree import initialize_aabbtree
from gpytoolbox.traverse_aabbtree import traverse_aabbtree

# This defines the functions needed to do a depth-first closest point traversal
class ray_mesh_intersect_traversal:
    def __init__(self,cam_pos,cam_dir,V,F):
        self.cam_pos = cam_pos
        self.cam_dir = cam_dir
        self.valid_divide_by_ind = np.argmax(np.abs(self.cam_dir))
        self.V = V
        self.F = F
        self.dim = V.shape[1]
        self.t = np.Inf
        self.id = -1
        self.lmbd = np.array([0,0,0])
    def traversal_function(self,q,C,W,CH,tri_indices,is_leaf):        
        # Distance is L1 norm of ptest minus center 
        if is_leaf:
            # print("Point:",self.ptest)
            # print("Verts:",self.V)
            # print("Element:",self.F[tri_indices[q],:])
            v0 = self.V[self.F[tri_indices[q],0],:]
            v1 = self.V[self.F[tri_indices[q],1],:]
            v2 = self.V[self.F[tri_indices[q],2],:]
            t,is_hit,hit_coord = ray_triangle_intersect(self.cam_pos,self.cam_dir,v0,v1,v2)
            # print("Distance",sqrD)
        else:
            center = C[q,:]
            width = W[q,:]
            is_hit,hit_coord = ray_box_intersect(self.cam_pos,self.cam_dir,center,width)
            t = ((hit_coord[self.valid_divide_by_ind] - self.cam_pos[self.valid_divide_by_ind])/self.cam_dir[self.valid_divide_by_ind])
        if (is_hit  and (t<self.t)):
            if (is_leaf):
                self.t = t
                b = barycentric_coordinates(hit_coord,v0,v1,v2)
                self.lmbd = b
                self.id = tri_indices[q] 
            return True
        return False
    def add_to_queue(self,queue,new_ind):
        # Depth first: insert at beginning (much less queries).
        queue.insert(0,new_ind)



def ray_mesh_intersect(cam_pos,cam_dir,V,F,use_embree=True,C=None,W=None,CH=None,tri_ind=None):
    """Shoot a ray from a position and see where it crashes into a given mesh

    Uses a bounding volume hierarchy to efficiently compute intersections of many different rays with a given mesh.

    Parameters
    ----------
    cam_pos : numpy double array
        Matrix of camera positions
    cam_dir : numpy double array
        Matrix of camera directions
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh
    use_embree : bool, optional (default True)
        Whether to use the much more optimzed C++ AABB embree implementation of ray mesh intersections. If False, uses gpytoolbox's native AABB tree and gpytoolbox's intersection queries.
    C : numpy double array, optional (default None)
        Matrix of AABB box centers (if None and use_embree=False, will be computed)
    W : numpy double array, optional (default None)
        Matrix of AABB box widths (if None and use_embree=False, will be computed)
    CH : numpy int array, optional (default None)
        Matrix of child indeces (-1 if leaf node). If None and use_embree=False, will be computed
    tri_indices : numpy int array, optional (default None)
        Vector of AABB element indices (-1 if *not* leaf node). If None and use_embree=False, will be computed

    Returns
    -------
    ts : list of doubles
        The i-th entry of this list is the time it takes a ray starting at the i-th camera position with the i-th camera direction to hit the surface (inf if no hit is detected)
    ids : list of ints
        The i-th entry is the index into F of the mesh element that the i-th ray hits (-1 if no hit is detected)
    lambdas : numpy double array
        The i-th row contains the barycentric coordinates of where in the triangle the ray hit (all zeros is no hit is detected)

    Examples
    --------
    ```python
    from gpytoolbox import ray_mesh_intersect
    v,f = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
    cam_pos = np.array([[1,0.1,0.1],[1,0.2,0.0]])
    cam_dir = np.array([[-1,0,0],[-1,0,0]])
    t, ids, l = ray_mesh_intersect(cam_pos,cam_dir,v,f)
    ```
    """
    if use_embree:
        try:
            from gpytoolbox_bindings import _ray_mesh_intersect_cpp_impl
        except:
            raise ImportError("Gpytoolbox cannot import its C++ decimate binding.")

        ts, ids, lambdas = _ray_mesh_intersect_cpp_impl(cam_pos.astype(np.float64),cam_dir.astype(np.float64),V.astype(np.float64),F.astype(np.int32))
    else:
        ts = np.Inf*np.ones(cam_pos.shape[0])
        ids = -np.ones(cam_pos.shape[0],dtype=int)
        lambdas = np.zeros((cam_pos.shape[0],3))
        # print("building tree")
        if ((C is None) or (W is None) or (tri_ind is None) or (CH is None)):
            C,W,CH,_,_,tri_ind = initialize_aabbtree(V,F=F)
        # print("built tree")
        # print("computing distances")
        for i in range(cam_pos.shape[0]):
            trav = ray_mesh_intersect_traversal(cam_pos[i,:],cam_dir[i,:],V,F)
            traverse_fun = trav.traversal_function
            add_to_queue_fun = trav.add_to_queue
            _ = traverse_aabbtree(C,W,CH,tri_ind,traverse_fun,add_to_queue=add_to_queue_fun)
            ts[i] = trav.t
            ids[i] = trav.id
            lambdas[i,:] = trav.lmbd
        # print("computed distances")
    return ts, ids, lambdas
