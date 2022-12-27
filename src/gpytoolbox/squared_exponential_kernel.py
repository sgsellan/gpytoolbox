import numpy as np

def squared_exponential_kernel(X1,X2,length=0.2,scale=1,derivatives=(-1,-1)):
    """
    Evaluates the squared exponential (i.e., gaussian density) kernel and its derivatives at any given pairs of points X1 and X2.

    Parameters
    ----------
    X1 : (num_points,dim) numpy array
        The first set of points to evaluate the kernel at.
    X2 : (num_points,dim) numpy array
        The second set of points to evaluate the kernel at.
    length : float
        The scalar length scale of the kernel.
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
    ```python
    # We can evaluate the kernel directly
    x = np.reshape(np.linspace(-1,1,num_samples),(-1,1))
    y = x + 0.2
    v = gpytoolbox.squared_exponential_kernel(x,y)
    # But more often we'll use it to call a gaussian process:
    def ker_fun(X1,X2,derivatives=(-1,-1)):
        return gpytoolbox.squared_exponential_kernel(X1,X2,derivatives=derivatives,length=1,scale=0.1)
    # Then, call a gaussian process
    x_train = np.linspace(0,1,20)
    y_train = 2*x_train
    x_test = np.linspace(0,1,120)
    y_test = gpytoolbox.gaussian_process(X_train,y_train,x_test,kernel=ker_fun)
    """
    indi = derivatives[0]
    indj = derivatives[1]
    gamma = 1/(length**2.0)
    if (indi==-1 and indj==-1):
        # This is just evaluating the kernel
        return scale*np.exp(-0.5*np.sum(((X1-X2) * (X1-X2)),axis=1)/(length**2.0))

    elif (indj==-1 or indi==-1):
        # Derivative wrt ind
        ind = np.maximum(indi,indj)
        sgn = np.sign(indi-indj) # Positive if indi is the nonzero one, negative otherwise
        return -sgn*gamma*(X1[:,ind] - X2[:,ind])*squared_exponential_kernel(X1,X2,length=length,scale=scale,derivatives=(-1,-1))
    else:
        # Derivative wrt indi and indj
        return ( (indi==indj)*gamma - gamma*gamma*(X1[:,indi] - X2[:,indi])*(X1[:,indj] - X2[:,indj]))*squared_exponential_kernel(X1,X2,length=length,scale=scale,derivatives=(-1,-1))
