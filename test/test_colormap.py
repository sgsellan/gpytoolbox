from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestColormap(unittest.TestCase):

    def test_colormaps(self):
        maps = ['BuGn','BuPu','GnBu','OrRd','PuBu','PuBuGn','PuRd','RdPu',
        'YlGn','YlGnBu','YlOrBr','YlOrRd','Blues','Greens','Greys','Oranges',
        'Purples','Reds','BrBG','PiYG','PRGn','PuOr','RdBu','RdGy','RdYlBu',
        'RdYlGn','Spectral','Accent','Dark2','Paired','Pastel1','Pastel2',
        'Set1','Set2','Set3']
        nums = [1,2,3,4,5,6,7,8]
        for map in maps:
            for num in nums:
                C = gpy.colormap(map, num, interpolate=False)
                self.assertTrue(C.shape == (num,3))
        for map in maps:
            # Check that with interpolate we can go well above the number of colors
            for num in range(1,500):
                C = gpy.colormap(map, num, interpolate=True)
                self.assertTrue(C.shape == (num,3))

if __name__ == '__main__':
    unittest.main()