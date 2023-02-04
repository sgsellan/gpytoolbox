# import gpytoolbox_pybind as gpy_pybind
import numpy as np
import gpytoolbox as gpy
from gpytoolbox_bindings import read_obj_pybind, viewer
# from gpytoolbox.gpytoolbox_bindings import read_obj_pybind
[n1, V, F, n2, n3, n4, n5] = read_obj_pybind("./unit_tests_data/armadillo.obj", False, False)


#test constructor
v = viewer()

v.background_color(np.array([0.8,1,1, 1.0]))
#test set_mesh
v.set_mesh(V- np.array([200, 0, 0]), F)
v.set_colors(np.array([0,0,1]))


#test append mesh with
id1 = v.append_mesh(V*4 , F) #id should be=1
v.set_colors(np.array([0,1,0]), id1)
show_faces = False
v.show_faces(show_faces, id1)

#test  append_mesh with nothing
id2 = v.append_mesh() #id should be=2
V2 = V*4 + np.array([400, 0, 0])
v.set_mesh(V2, F, id2)
d = V[:, 0]
v.set_data(d, id=id2, colormap="jet")
show_lines = False
v.show_lines( show_lines, id=id2 )


#launch
v.launch()
