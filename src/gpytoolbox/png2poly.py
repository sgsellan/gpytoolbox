import numpy as np
from skimage import measure
from skimage.color.colorconv import rgb2gray, rgba2rgb
from imageio import imread

def png2poly(filename):
    # Reads a png file and outputs a list of polylines that constitute
    # the contours of the png, using marching squares. This is useful for 
    # generating 2D "realworld" data. 
    # 
    # This follows code under MIT licence by Stephan Hügel.
    #
    # Input:
    #       filename string containing the path (relative or absolute) to the 
    #           png file 
    #
    # Output:
    #       poly  a list where the i-th element is a #Vi by 2 polyline 
    #
    # Note: This often results in "duplicate" polylines (one is the 
    # white->black contour, other is the black->white contour.
    # 
    polypic = imread(filename)
   
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