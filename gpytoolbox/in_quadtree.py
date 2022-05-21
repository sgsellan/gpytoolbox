import numpy as np

def in_quadtree(point,C,W,CH):
    # IN_QUADTREE
    # Traverses a quadtree in logarithmic time to find the smallest cell
    # containing a given point in 2D space
    #
    # i, others = in_quadtree(point,C,W,CH)
    #
    # Inputs:
    #   point size-3 vector of point in the plane
    #   C #nodes by 3 matrix of cell centers
    #   W #nodes vector of cell widths (**not** half widths)
    #   CH #nodes by 4 matrix of child indeces (-1 if leaf node)
    #
    # Outputs:
    #   i integer index of smallest cell containint P into C,W,CH
    #   others vector of integers to all other (non-leaf) cells containing P

    others = []
    queue = [0]
    i = -1 # by default it's nowhere
    dim = C.shape[1]
    
    while len(queue)>0:
        # Pop from queue
        q = queue.pop(0)
        # Check if point is inside this cell
        if is_in_quad(point[None,:],C[q,:],W[q]):
            # If inside this cell, then add to otthers
            others.append(q)
            # Is it child?
            is_child = (CH[q,1]==-1)
            if is_child:
                # If it is, then we're done
                i = q
                break
            else:
                # If not, add children to queue
                queue.append(CH[q,0])
                queue.append(CH[q,1])
                queue.append(CH[q,2])
                queue.append(CH[q,3])
                if dim==3:
                    queue.append(CH[q,4])
                    queue.append(CH[q,5])
                    queue.append(CH[q,6])
                    queue.append(CH[q,7])
    return i, others
            
        
    
    
    




def is_in_quad(queries,center,width):
    dim = queries.shape[1]
    max_corner = center + width*np.tile(np.array([0.5]),dim)
    min_corner = center - width*np.tile(np.array([0.5]),dim)
    b = np.ones(queries.shape[0],dtype=bool)
    for dd in range(dim):
        b = (b & (queries[:,dd]>=min_corner[dd]) & (queries[:,dd]<=max_corner[dd]))
    return b