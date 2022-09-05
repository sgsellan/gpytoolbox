import numpy as np
from gpytoolbox.boundary_edges import boundary_edges


def boundary_loops(f, allow_wrong_orientations=True):
    """Computes oriented boundary loop for each boundary component of a triangle
    mesh.
    This function only works on manifold triangle meshes.

    TODO: assert manifoldness when running this function

    Parameters
    ----------
    f : (m,3) numpy int array
        face index list of a triangle mesh
    allow_wrong_orientations: bool, optional (default True).
        whether to allow F to contain wrongly oriented triangles

    Returns
    -------
    loops : list of numpy arrays that are themselves lists of boundary vertices
        in oriented loops

    Examples
    --------
    TODO
    
    """

    assert f.shape[0] > 0
    assert f.shape[1] == 3

    bE = boundary_edges(f)

    #Loop through each boundary, edge, marking them as seen, until all have
    # been seen.
    unseen = np.full(bE.shape[0], True)
    loops = []
    while np.any(unseen):
        current_b = np.argmax(unseen)
        current_bE = bE[current_b,:]
        start = current_bE[0]
        unseen[current_b] = False

        loop_vertices = []
        loop_vertices.append(start)

        head = current_bE[1]
        while head != start:
            loop_vertices.append(head)

            if allow_wrong_orientations:
                current_b_0 = np.where((bE[:,0] == head) & unseen)[0]
                current_b_1 = np.where((bE[:,1] == head) & unseen)[0]
                if len(current_b_0)>len(current_b_1):
                    current_b = current_b_0
                    head_ind = 1
                else:
                    current_b = current_b_1
                    head_ind = 0
            else:
                current_b = np.where((bE[:,0] == head) & unseen)[0]
                head_ind = 1
            assert len(current_b) == 1
            current_b = current_b.item(0)
            current_bE = bE[current_b,:]
            unseen[current_b] = False
            head = current_bE[head_ind]

        loops.append(np.array(loop_vertices))

    assert sum([len(l) for l in loops]) == bE.shape[0]
    return loops
