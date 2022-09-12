import numpy as np

def compactly_supported_normal(x,n=2,sigma=1.,center=None,second_derivative=-1):
    """Approximated compact Gaussian density functiona
    
    This computes a compactly supported approximation of a Gaussian density  function by convolving a box filter with itself n times in each dimension and multiplying, as described in "Poisson Surface Reconstruction" by Kazhdan et al. 2006.

    Parameters
    ----------
    x : (nx,dim) numpy double array
        Coordinates where the function is to be evaluated
    n : int, optional (default 2)
        Number of convolutions (between 1 and 4), a higher number will more closely resemble a Gaussian but have broader support
    sigma : double, optional (default 1.0)
        Scale / standard deviation function (bigger leades to broader support)
    center: (dim,) numpy double array (default None)
        Coordinates where the function is centered (i.e., distribution mean). If None, center at the origin.
    second_derivative : int, optional (default -1)
        If greater or equal than zero, this is the index of the dimension along which to compute the second partial derivative instead of the pure density.

    Returns
    -------
    v : (nx,) numpy double array
        Values of compacly supported approximate Gaussian function at x

    Examples
    --------
    ```python
    from gpytoolbox import compactly_supported_normal
    import matplotlib.pyplot as plt
    x = np.reshape(np.linspace(-4,4,1000),(-1,1))
    plt.plot(x,compactly_supported_normal(x, n=4,center=np.array([0.5])))
    plt.plot(x,compactly_supported_normal(x, n=3,center=np.array([0.5])))
    plt.plot(x,compactly_supported_normal(x, n=2,center=np.array([0.5])))
    plt.plot(x,compactly_supported_normal(x, n=1,center=np.array([0.5])))
    plt.show()
    ```
    """

    ndim = x.ndim
    if ndim==0:
        x = np.array([[x]])
    if ndim==1:
        x = np.reshape(x,(-1,1))
    # print(ndim)
    dim = x.shape[1]
    
    if (center is None):
        center = np.zeros(dim)
    x_new = (x - np.tile(center[None,:],(x.shape[0],1)))/sigma

    # This is the function centered at 0 and with sigma=1
    def compactly_supported_normal_centered(x,n,second_derivative):
        vals = np.ones(x.shape[0])
        # I got this by writing into Wolfram Alpha:
        # Convolve[rect(y),rect(y)]
        # Convolve[rect(y),triangle(y)]
        # Convolve[rect(y),Piecewise[{{1/6 (-abs(y)^3 + 6 y^2 - 12 abs(y) + 8),1<=abs(y)<=2},{1/6 (3abs(y)^3 - 6y^2 + 4 ),abs(y)<1}}]]
        for i in range(dim):
            xx = x[:,i]
            if second_derivative==i:
                if n==2:
                    xx_val = (np.abs(xx)<0.5)*(-2.0) + (np.abs(xx)<1.5)*(np.abs(xx)>0.5)*(1.0)    
                elif n==1:
                    xx_val = 0.0*np.abs(xx)
                elif n==3:
                    xx_val = (np.abs(xx)<1)*(3*3*2*(np.abs(xx)) - 12)*(1/6)  +  (np.abs(xx)<=2)*(np.abs(xx)>=1)*( -3*2*(np.abs(xx)) + 12)*(1/6.)
                elif n==4:
                    xx_val = (np.abs(xx)<=2.5)*(np.abs(xx)>=1.5)*( 16*4*3*(np.abs(xx)**2.0) - 160*3*2*(np.abs(xx)) + 600*2)*(1/384) + (np.abs(xx)<=1.5)*(np.abs(xx)>=0.5)*( -16*4*3*(np.abs(xx)**2.0) + 80*3*2*(np.abs(xx)) - 120*2)*(1/96) + (np.abs(xx)<=0.5)*( 48*4*3*(np.abs(xx)**2.0) - 120*2)*(1/192)
            else:
                if n==2:
                    xx_val = (np.abs(xx)<0.5)*(-(xx**2.0)+0.75) + (np.abs(xx)<1.5)*(np.abs(xx)>0.5)*(0.5*(xx**2.0) - 1.5*np.abs(xx) + 1.125)
                elif n==1:
                    xx_val = (np.abs(xx)<1.0)*(1 - np.abs(xx))
                elif n==3:
                    xx_val = (np.abs(xx)<1)*(3*(np.abs(xx)**3.0) - 6*(np.abs(xx)**2.0) + 4)*(1/6)  +  (np.abs(xx)<=2)*(np.abs(xx)>=1)*( -(np.abs(xx)**3.0) + 6*(np.abs(xx)**2.0) - 12*np.abs(xx) + 8 )*(1/6.)
                elif n==4:
                    xx_val = (np.abs(xx)<=2.5)*(np.abs(xx)>=1.5)*( 16*(np.abs(xx)**4.0) - 160*(np.abs(xx)**3.0) + 600*(np.abs(xx)**2.0) - 1000*(np.abs(xx)) + 625)*(1/384) + (np.abs(xx)<=1.5)*(np.abs(xx)>=0.5)*( -16*(np.abs(xx)**4.0) + 80*(np.abs(xx)**3.0) - 120*(np.abs(xx)**2.0) + 20*(np.abs(xx)) + 55)*(1/96) + (np.abs(xx)<=0.5)*( 48*(np.abs(xx)**4.0) - 120*(np.abs(xx)**2.0) + 115)*(1/192)

            vals = vals * xx_val
        return vals

    val = compactly_supported_normal_centered(x_new,n,second_derivative)/(sigma**dim)
    if second_derivative>=0:
        val = val/(sigma**2.0)
    
    # We call it with the transformed x and rescale by sigma
    return val




