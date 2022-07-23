import numpy as np

def compactly_supported_normal(x,n=2,sigma=1.,center=None,second_derivative=-1):
    # This computes a compactly supported approximation of a Gaussian density 
    # function by convolving a box filter with itself n times.
    #
    # Inputs:
    #       x #x by dim array of #x inputs
    #       Optional:
    #           n integer number of convolutions (n-1: polynomial degree)
    #           sigma scalar scale factor (bigger is wider spread)
    #           center #dim numpy array of where the function is centered
    #           second_derivative index of dimension along which to compute
    #               the second partial derivative (by default, -1, no axis)
    #
    # Output:
    #       vals #x numpy array of values
    
    dim = x.shape[1]
    if (center is None):
        center = np.zeros(dim)
    x_new = (x - np.tile(center[None,:],(x.shape[0],1)))/sigma

    # This is the function centered at 0 and with sigma=1
    def compactly_supported_normal_centered(x,n,second_derivative):
        vals = np.ones(x.shape[0])
        # Write this into Wolfram Alpha:
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
                    xx_val = (np.abs(xx)<1)*(3**3*2*(np.abs(xx)) - 12)*(1/6)  +  (np.abs(xx)<=2)*(np.abs(xx)>=1)*( -3*2*(np.abs(xx)) + 12)*(1/6.)
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

    
    # We call it with the transformed x and rescale by sigma
    return compactly_supported_normal_centered(x_new,n,second_derivative)/(sigma**dim)




