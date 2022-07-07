from context import gpytoolbox
import numpy as np
import polyscope as ps

from gpytoolbox.in_quadtree import in_quadtree

np.random.seed(0)
th = 2*np.pi*np.random.rand(500,1)
P = 0.5*np.concatenate((np.cos(th),np.sin(th)),axis=1)

C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8,vmin=np.array([-1,-1]),vmax=np.array([1,1]))
V,Q,_ = gpytoolbox.bad_quad_mesh_from_quadtree(C,W,CH)

# Generate random query points 
queries = 2*np.random.rand(200,2)-1


ps.init()

i = 0
ps.register_surface_mesh("test quadtree",V,Q,edge_width=1)
query_point = ps.register_point_cloud("query point",queries[i,:][None,:])
ind, others = in_quadtree(queries[i,:],C,W,CH)
ind_cell_verts = np.vstack((
        C[ind,:][None,:] + 0.5*W[ind]*np.array([[-1,-1]]),
        C[ind,:][None,:] + 0.5*W[ind]*np.array([[1,-1]]),
        C[ind,:][None,:] + 0.5*W[ind]*np.array([[1,1]]),
        C[ind,:][None,:] + 0.5*W[ind]*np.array([[-1,1]])
))
ind_cell_edges = np.array([[0,1],[1,2],[2,3],[3,0]])
ps_net = ps.register_curve_network("cell", ind_cell_verts, ind_cell_edges)
ps.set_navigation_style("planar")

def callback():
    global i
    i = i + 1
    if i<200:
        ind, others = in_quadtree(queries[i,:],C,W,CH)
        ind_cell_verts = np.vstack((
            C[ind,:][None,:] + 0.5*W[ind]*np.array([[-1,-1]]),
            C[ind,:][None,:] + 0.5*W[ind]*np.array([[1,-1]]),
            C[ind,:][None,:] + 0.5*W[ind]*np.array([[1,1]]),
            C[ind,:][None,:] + 0.5*W[ind]*np.array([[-1,1]])
        ))
        ps_net.update_node_positions(ind_cell_verts)
        query_point.update_point_positions(queries[i,:][None,:])

ps.set_user_callback(callback)
ps.show()