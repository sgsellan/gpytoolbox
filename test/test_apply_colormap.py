from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestApplyColormap(unittest.TestCase):
    def test_apply_colormaps(self):
        maps = ['BuGn','BuPu','GnBu','OrRd','PuBu','PuBuGn','PuRd','RdPu',
        'YlGn','YlGnBu','YlOrBr','YlOrRd','Blues','Greens','Greys','Oranges',
        'Purples','Reds','BrBG','PiYG','PRGn','PuOr','RdBu','RdGy','RdYlBu',
        'RdYlGn','Spectral','Accent','Dark2','Paired','Pastel1','Pastel2',
        'Set1','Set2','Set3']
        for map in maps:
            for n in range(1,150):
                C = gpy.apply_colormap(gpy.colormap(map, 60),
                    np.random.default_rng().random(n))
                self.assertTrue(C.shape == (n,3))
            # If equally spaced, should be same as colormap matrix
            for num_colors in range(10,60):
                C = gpy.apply_colormap(gpy.colormap(map, num_colors),
                    np.linspace(0,1,num_colors))
                self.assertTrue((C==np.round(gpy.colormap(map, num_colors)).astype(np.int32)).all())


if __name__ == '__main__':
    unittest.main()