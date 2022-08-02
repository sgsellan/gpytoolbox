from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestWritePly(unittest.TestCase):
    # In the future, it would be nice to have a test where we read and then write a function and check that it's the same. But we don't have a read_ply yet. So, for now, a tiny test we can do is that at least it accepts all the arguments it claims to accept.
    def test_write_parameters(self):
        meshes = ["bunny_oded.obj", "armadillo.obj", "armadillo_with_tex_and_normal.obj", "bunny.obj", "mountain.obj"]
        maps = ['BuGn','BuPu','GnBu','OrRd','PuBu','PuBuGn','PuRd','RdPu',
        'YlGn','YlGnBu','YlOrBr','YlOrRd','Blues','Greens','Greys','Oranges',
        'Purples','Reds','BrBG','PiYG','PRGn','PuOr','RdBu','RdGy','RdYlBu',
        'RdYlGn','Spectral','Accent','Dark2','Paired','Pastel1','Pastel2',
        'Set1','Set2','Set3']
        for map in maps:
            for mesh in meshes:
                V,F,_,_,_,_ = \
                gpy.read_mesh("test/unit_tests_data/" + mesh,
                    return_UV=True, return_N=True, reader='C++')
                gpy.write_ply("test/unit_tests_data/temp.obj",V,faces=F)
                gpy.write_ply("test/unit_tests_data/temp.obj",V,faces=F,colors=np.random.rand(V.shape[0]))
                gpy.write_ply("test/unit_tests_data/temp.obj",V,faces=F,colors=np.random.rand(V.shape[0]),cmap=map)

if __name__ == '__main__':
    unittest.main()
