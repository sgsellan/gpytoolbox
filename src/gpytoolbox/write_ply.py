import numpy as np
from gpytoolbox.colormap import colormap
from gpytoolbox.apply_colormap import apply_colormap

def write_ply(filename,vertices,faces=None,colors=None,cmap='BuGn'):
    """Store triangle mesh into .ply file format
    
    Writes a triangle mesh (optionally with per-vertex colors) into the ply file format, in a way consistent with, e.g., importing to Blender.

    Parameters
    ----------
    filename : str
        Name of the file ending in ".ply" to which to write the mesh
    vertices : numpy double array
        Matrix of mesh vertex coordinates
    faces : numpy int array, optional (default None)
        Matrix of triangle face indices into vertices. If none, only the vertices will be written (e.g., a point cloud)
    colors : numpy double array, optional (default None)
        Array of per-vertex colors. It can be a matrix of per-row RGB values, or a vector of scalar values that gets transformed by a colormap.
    cmap : str, optional (default 'BuGn')
        Name of colormap used to transform the color values if they are a vector of scalar function values (if colors is a matrix of RGB values, this parameter will not be used). Should be a valid input to `colormap`.

    See Also
    --------
    write_mesh, colormap.

    Notes
    -----
    This function is not optimized and covers the very specific funcionality of saving a mesh with per-vertex coloring that can be imported into Blender or other software. If you wish to write a mesh for any other purpose, we strongly recommend you use write_mesh instead.

    Examples
    --------
    TODO
    """

    vertices = vertices.astype(float)
    f = open(filename,"w")
    f.write("ply\nformat {} 1.0\n".format('ascii'))
    f.write("element vertex {}\n".format(vertices.shape[0]))
    f.write("property double x\n")
    f.write("property double y\n")
    f.write("property double z\n")
    if (colors is not None):
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
    if (faces is not None):
        f.write("element face {}\n".format(faces.shape[0]))
    else:
        f.write("element face 0\n")
    f.write("property list int int vertex_indices\n")
    f.write("end_header\n")
    # write_vert_str = "{} {} {}\n" * vertices.shape[0]
    # f.write(write_vert_str.format(tuple(np.reshape(vertices,(-1,1)))))
    # This for loop should be vectorized
    if (colors is None):
        for i in range(vertices.shape[0]):
            f.write("{} {} {}\n".format(vertices[i,0],vertices[i,1],vertices[i,2]))
    else:
        if (colors.ndim==1 or colors.shape[1]==1): # color is scalar values
            C = apply_colormap(colormap(cmap, 200), colors)
        else:
            if np.max(colors)<=1:
                C = np.round(colors*255)
            else:
                C = colors
        # This should be vectorized
        for i in range(vertices.shape[0]):
            f.write("{} {} {} {} {} {} 255\n".format(vertices[i,0],vertices[i,1],vertices[i,2],int(C[i,0]),int(C[i,1]),int(C[i,2])))
    # This should be vectorized
    if (faces is not None):
        for i in range(faces.shape[0]):
            f.write("3 {} {} {}\n".format(faces[i,0],faces[i,1],faces[i,2]))
    f.close()