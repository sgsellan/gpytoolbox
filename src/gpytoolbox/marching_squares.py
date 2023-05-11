import numpy as np
from .remove_duplicate_vertices import remove_duplicate_vertices

def marching_squares(S,GV,nx,ny):
    """
    Marching squares algorithm for extracting isocontours from a scalar field.
    S: scalar field
    nx,ny: number of grid points in x and y direction
    """
    S = np.reshape(S,(nx,ny),order='F')
    # Create empty list for
    verts = []
    edge_list = [] # index of edge vertices
    # Loop over all grid points
    for i in range(nx-1):
        for j in range(ny-1):
            # Get the scalar values at the corners of the grid cell
            a = S[i,j]
            b = S[i+1,j]
            c = S[i+1,j+1]
            d = S[i,j+1]
            # Get the contour index
            k = 0
            if a > 0:
                k += 1
            if b > 0:
                k += 2
            if c > 0:
                k += 4
            if d > 0:
                k += 8
            # Use symmetry
            flip = False
            if k > 7:
                flip = True
                k = 15 - k
            
            # Get the contour line segments
            if k == 1:
                # x = i
                x = i - a/(b-a)
                y = j
                verts.append([x,y])
                x = i
                y = j - a/(d-a)
                # y = j
                verts.append([x,y])
                if flip:
                    edge_list.append([len(verts)-2,len(verts)-1])
                else:
                    edge_list.append([len(verts)-1,len(verts)-2])
            elif k == 2:
                x = i - a/(b-a)
                # x = i
                y = j
                verts.append([x,y])
                x = i + 1
                y = j - b/(c-b)
                verts.append([x,y])
                if flip:
                    edge_list.append([len(verts)-1,len(verts)-2])
                else:
                    edge_list.append([len(verts)-2,len(verts)-1])
            elif k == 3:
                x = i
                y = j - a/(d-a)
                verts.append([x,y])
                x = i + 1
                y = j - b/(c-b)
                verts.append([x,y])
                if flip:
                    edge_list.append([len(verts)-1,len(verts)-2])
                else:
                    edge_list.append([len(verts)-2,len(verts)-1])
            elif k == 4:
                x = i + 1
                y = j- b/(c-b)
                verts.append([x,y])
                x = i - d/(c-d)
                y = j + 1
                verts.append([x,y])
                if flip:
                    edge_list.append([len(verts)-1,len(verts)-2])
                else:
                    edge_list.append([len(verts)-2,len(verts)-1])
            elif k == 5:
                x = i - a/(b-a)
                y = j
                verts.append([x,y])
                x = i
                y = j - a/(d-a)
                # y = j
                verts.append([x,y])
                x = i + 1
                y = j- b/(c-b)
                verts.append([x,y])
                x = i - d/(c-d)
                y = j + 1
                verts.append([x,y])
                if flip:
                    edge_list.append([len(verts)-4,len(verts)-3])
                else:
                    edge_list.append([len(verts)-3,len(verts)-4])
                if flip:
                    edge_list.append([len(verts)-1,len(verts)-2])
                else:
                    edge_list.append([len(verts)-2,len(verts)-1])
            elif k == 6:
                x = i - a/(b-a)
                y = j
                verts.append([x,y])
                x = i - d/(c-d)
                y = j + 1
                verts.append([x,y])
                if flip:
                    edge_list.append([len(verts)-1,len(verts)-2])
                else:
                    edge_list.append([len(verts)-2,len(verts)-1])
            elif k == 7:
                x = i - d/(c-d)
                y = j + 1
                verts.append([x,y])
                x = i
                y = j - a/(d-a)
                verts.append([x,y])
                if flip:
                    edge_list.append([len(verts)-2,len(verts)-1])
                else:
                    edge_list.append([len(verts)-1,len(verts)-2])
            else:
                pass

    # Convert list to numpy array
    verts = np.array(verts)
    edges = np.array(edge_list)
    verts, SVI, SVJ, edges = remove_duplicate_vertices(verts,faces=edges,
        epsilon=np.sqrt(np.finfo(verts.dtype).eps))

    # Remove trivial edges
    edges = edges[np.not_equal(edges[:,0], edges[:,1]), :]

    # # Remove duplicate edges
    # edges = np.unique(edges, axis=0)

    # Rescale to original grid
    verts[:,0] = verts[:,0]/(nx-1)
    verts[:,1] = verts[:,1]/(ny-1)
    verts = verts*(GV.max(axis=0)-GV.min(axis=0)) + GV.min(axis=0)

    return verts, edges
