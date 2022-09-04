import numpy as np
from skimage import measure
from skimage.color.colorconv import rgb2gray, rgba2rgb
from skimage.io import imread
from skimage.transform import rotate

def png2poly(filename):
    """Export polylines from png image
    
    Reads a png file and outputs a list of polylines that constitute the contours of the png, using marching squares. This is useful for generating 2D "realworld" data. 

    Parameters
    ----------
    filename : str
        Path to png file
    
    Returns
    -------
    poly : list of numpy double arrays
        Each list element is a matrix of ordered polyline vertex coordinates

    Notes
    -----
    This often results in "duplicate" polylines (one is the white->black contour, other is the black->white contour.

    Examples
    --------
    TODO
    """
    polypic = imread(filename)
    # For some reason reading the image flips it by 90 degrees. This fixes it
    polypic = rotate(polypic, angle=-90, resize=True)
    # print(polypic)
   
    # convert to greyscale and remove alpha if neccessary
    if len(polypic.shape)>2:
        if polypic.shape[2]==4:
            polypic = rgb2gray(rgba2rgb(polypic))
        elif polypic.shape[2]==3:
            polypic = rgb2gray(polypic)
    # find contours
    polypic = polypic/np.max(polypic)
    poly = measure.find_contours(polypic, 0.5)
    return poly