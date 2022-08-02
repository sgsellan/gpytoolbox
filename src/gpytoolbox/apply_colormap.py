import numpy as np
import scipy as sp


def apply_colormap(C, f,
       min=None,
       max=None,
       log=False,
       piecewise_linear=False):
    """Maps function values to colors of a given colormap.
    
    Apply a colormap produced with the colormap function to an array of
    scalar values.
    
    Parameters
    ----------
    C : numpy int array
        Colormap matrix where each row corresponds to an RGB color. Compatible with maps produced by colormap
    f : numpy double array
        Vector of scalar function values to which to apply the colormap
    min : double, optional (default None)
        Scalar value corresponding to the beginning of C. Will auto-scale if None.
    max : double, optional (default None)
        Scalar value corresponding to the end of C. Will auto-scale if None.
    log : bool, optional (default False)
        Whether to log-scale the colormap or not. For this option, all values will be clamped to larger than machine epsilon.
    piecewise_linear : bool, optional (default False)
        Whether to treat the colormap as piecewise linear

    Returns
    -------
    f_colored : numpy double array
        Matrix of C applied to f. Each row corresponds to a scalar value in f.

    See Also
    --------
    colormap.

    Notes
    -----
    All colormaps are assumed to be piecewise constant unless otherwise
    specified (be aware: the first and last colors of the colormap will span
    half as much length as the other colors, since they represent the colors
    at the minimum and maximum respectively).


    Examples
    --------
    TODO
    """

    assert C.shape[0] >= 1
    assert C.shape[1] == 3

    if C.shape[0] == 1:
        return np.repeat(C, f.shape[0], axis=0)

    interpolation_mode = 'linear' if piecewise_linear else 'nearest'

    minimum = np.amin(f) if min is None else min
    if log and minimum < np.finfo(np.float64).eps:
        minimum = np.finfo(np.float64).eps
    maximum = np.amax(f) if max is None else max
    assert maximum >= minimum
    if maximum < minimum + np.finfo(np.float64).eps:
        maximum = minimum + np.finfo(np.float64).eps

    f = np.minimum(np.maximum(f, minimum), maximum)

    if log:
        xaxis = np.logspace(np.log2(minimum), np.log2(maximum), C.shape[0],
              base=2.)
    else:
        xaxis = np.linspace(minimum, maximum, C.shape[0])

    interpolator = sp.interpolate.interp1d(xaxis, C, axis=0,
        kind=interpolation_mode)
    return interpolator(f).round().astype(int)
