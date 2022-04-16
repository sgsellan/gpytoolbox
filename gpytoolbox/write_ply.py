import numpy as np
import matplotlib.pyplot as plt

def write_ply(filename,vertices,faces,colors=None):
    # Writes a triangle mesh with colors into ply file format,
    # in a way consistent with, e.g., importing to Blender 
    # 
    # Input:
    #       filename a string of the full path where to write the mesh
    #               (will be overwritten if exists)
    #       V #V by 3 numpy array of mesh vertex positions
    #       F #F by 3 numpy array of mesh face indeces on V
    #
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
    f.write("element face {}\n".format(faces.shape[0]))
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
        # to-do: make this different
            colors = plt.cm.viridis((np.clip(colors,np.min(colors),np.max(colors))-np.min(colors))/(np.max(colors) - np.min(colors)))
        if np.max(colors)<=1:
            C = np.round(colors*255)
        else:
            C = colors
        # This should be vectorized
        for i in range(vertices.shape[0]):
            f.write("{} {} {} {} {} {} 255\n".format(vertices[i,0],vertices[i,1],vertices[i,2],int(C[i,0]),int(C[i,1]),int(C[i,2])))
    # This should be vectorized
    for i in range(faces.shape[0]):
        f.write("3 {} {} {}\n".format(faces[i,0],faces[i,1],faces[i,2]))
    f.close()