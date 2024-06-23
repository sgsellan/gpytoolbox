import numpy as np

def regular_circle_polyline(n):
    """Create a circle polyline

    Generates a regular polyline of a circle with radius 1 centered at the origin.

    Parameters
    ----------
    n : int
        number of vertices on the circle. Must be at least 3.

    Returns
    -------
    V : numpy double array
        Matrix of triangle polyline vertex coordinates
    E : numpy int array
        Matrix of edge vertex indices into V

    See Also
    --------
    regular_square_mesh.

    Examples
    --------
    ```python
    # Generate a polyline with n vertices
    n = 10
    V, E = gpytoolbox.regular_circle_polyline(n)
    ```
    """

    assert n>=3, "At least 3 vertices on the boundary."

    φ = np.linspace(0., 2.*np.pi, n)
    x = np.cos(φ)
    y = np.sin(φ)
    V = np.stack((x,y), axis=-1)

    E = np.stack((
        np.arange(0,n),
        np.concatenate((np.arange(1,n),[0]))
        ), axis=-1)

    return V,E
