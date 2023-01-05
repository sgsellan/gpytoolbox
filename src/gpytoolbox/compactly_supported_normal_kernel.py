import numpy as np

def compactly_supported_normal_kernel(X1,X2,length=0.2,scale=1,derivatives=(-1,-1)):
    """
    Evaluates the squared exponential (i.e., gaussian density) kernel and its derivatives at any given pairs of points X1 and X2.

    Parameters
    ----------
    X1 : (num_points,dim) numpy array
        The first set of points to evaluate the kernel at.
    X2 : (num_points,dim) numpy array
        The second set of points to evaluate the kernel at.
    length : float or (num_points,) numpy array
        The scalar length scale of the kernel (can be a vector if the length is different for different points).
    scale : float
        The scalar scale factor of the kernel.
    derivatives : (2,) numpy array
        The indices (i,j) of the derivatives to take: d^2 K / dx1_i dx2_j. A value of -1 means no derivative. Note that these are not commutative: the derivative wrt x1_i is not the same as the derivative wrt x2_i (there's a minus sign difference).

    Returns
    -------
    K : (num_points,) numpy array
        The kernel evaluated at the given points.
    
    Examples
    --------
    TODO
    
    """
    r = X1 - X2
    ndim = r.ndim
    if ndim==0:
        r = np.array([[r]])
    if ndim==1:
        r = np.reshape(r,(-1,1))

    
    # print(ndim)
    dim = r.shape[1]

    if (np.isscalar(length)):
        length = np.ones(r.shape[0])*length
    
    
    center = np.zeros(dim)
    r_new = (r - np.tile(center[None,:],(r.shape[0],1)))/np.tile(length[:,None],(1,dim))

    # This is the function centered at 0 and with sigma=1
    def compactly_supported_normal_kernel_centered(x,derivatives=(-1,-1)):
        vals = np.ones(x.shape[0])
        # I got this by writing into Wolfram Alpha:
        # Convolve[rect(y),rect(y)]
        # Convolve[rect(y),triangle(y)]
        # Convolve[rect(y),Piecewise[{{1/6 (-abs(y)^3 + 6 y^2 - 12 abs(y) + 8),1<=abs(y)<=2},{1/6 (3abs(y)^3 - 6y^2 + 4 ),abs(y)<1}}]]
        for i in range(dim):
            xx = x[:,i]
            sign = (xx>=0)*1.0 + (xx<0)*-1.0
            if (derivatives[0]!=i and derivatives[1]!=i):
                # Then no derivative in this dimension,
                xx_val = (np.abs(xx)<1)*(3*(np.abs(xx)**3.0) - 6*(np.abs(xx)**2.0) + 4)*(1/6)  +  (np.abs(xx)<=2)*(np.abs(xx)>=1)*( -(np.abs(xx)**3.0) + 6*(np.abs(xx)**2.0) - 12*np.abs(xx) + 8 )*(1/6.)
            elif (derivatives[0]==i and derivatives[1]!=i):
                # Only one derivative, and positive              
                xx_val = (np.abs(xx)<1)*( 9*sign*(np.abs(xx)**2.0) - 12*xx)*(1/6.)  +  (np.abs(xx)<=2)*(np.abs(xx)>=1)*( -3*sign*(xx**2.0) + 12*xx - 12*sign)*(1/6.)
            elif (derivatives[0]!=i and derivatives[1]==i):
                xx_val = (np.abs(xx)<1)*( 9*sign*(np.abs(xx)**2.0) - 12*xx)*(1/6.)  +  (np.abs(xx)<=2)*(np.abs(xx)>=1)*( -3*sign*(xx**2.0) + 12*xx - 12*sign)*(1/6.)
                xx_val = -xx_val
                # Only one derivative, and negative
            elif (derivatives[0]==i and derivatives[1]==i):
                # Two derivatives, and positive
                # TODO: doublecheck minus sign!
                xx_val = (np.abs(xx)<1)*(3*3*2*(np.abs(xx)) - 12)*(1/6)  +  (np.abs(xx)<=2)*(np.abs(xx)>=1)*( -3*2*(np.abs(xx)) + 12)*(1/6.)
                xx_val = -xx_val
            vals = vals * xx_val
        return vals

    val = compactly_supported_normal_kernel_centered(r_new,derivatives=derivatives)/(length**dim)
    val = val*scale
    if (derivatives[0]==-1 and derivatives[1]==-1):
        return val
    elif (derivatives[0]==-1 or derivatives[1]==-1):
        val = val/length
        return val
    else:
        val = val/(length**2.0)
        return val


