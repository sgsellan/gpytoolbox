def quadtree_boundary(CH,A):
    # Returns indeces of cells that fall on the boundary (defined as cells that
    # have no neighbor in at least one direction)
    #
    # Inputs:
    #   CH #nodes by 4 matrix of child indeces (-1 if leaf node)
    #   A #nodes by #nodes sparse adjacency matrix, where a value of a in the
    #       (i,j) entry means that node j is to the a-th direction of i
    #       (a=1: left;  a=2: right;  a=3: bottom;  a=4: top).
    #
    # Outputs:
    #   children_boundary is a list of indeces to CH and A of cells that are 
    #       both boundary cells and leaf nodes in the tree
    #   other_boundary is a list of all boundary cells regardless of parenthood 
    #
    #
    #

    children_boundary = []
    other_boundary = []
    for i in range(A.shape[0]):
        # go over possible direction
        for j in range(1,5):
            # is there a neighbor in this direction?
            j_neighbors = (A[i,:]==j).sum(axis=1).squeeze()
            #print(j_neighbors)
            if j_neighbors==0:
                other_boundary.append(i)
                if CH[i,0]==-1:
                    children_boundary.append(i)
                break

    # # Get uniques
    # children_boundary = list(set(children_boundary))
    # other_boundary = list(set(other_boundary))
    return children_boundary, other_boundary
    
    