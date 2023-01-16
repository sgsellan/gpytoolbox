import numpy as np




def quadtree_children(CH):
    """"Leaf nodes of a quadtree

    Builds a list of indeces to the child cells of the quadtree
    
    Parameters
    ----------
    CH : numpy int array
        Matrix of child indeces (-1 if leaf node)

    Returns
    -------
    child_indeces : list
        Indeces into CH and A of leaf node cells

    See Also
    --------
    initialize_quadtree, quadtree_boundary.

    Examples
    --------
    ```python
    # Create a random point cloud
    P = 2*np.random.rand(100,2) - 1
    # Initialize the quadtree
    C,W,CH,PAR,D,A = gpytoolbox.initialize_quadtree(P,graded=True,max_depth=8)
    # Get the children
    children = gpytoolbox.quadtree_children(CH)
    ```  
    """
    # Builds a list of indeces to the child cells of the quadtree
    #
    # Inputs:
    #   CH #nodes by 4 matrix of child indeces (-1 if leaf node)
    #
    # Outputs:
    #   child_indeces list of child indeces

    return (CH[:,0]<0).nonzero()[0]