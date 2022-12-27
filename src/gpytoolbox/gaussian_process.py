import numpy as np
from scipy.sparse import csc_matrix, eye, vstack, hstack
from scipy.sparse.linalg import splu, cg, cg
from .matrix_from_function import matrix_from_function
from .squared_exponential_kernel import squared_exponential_kernel

def gaussian_process(X_train,y_train,X_test,kernel=None,X_induced=None,grad_y_train=None,verbose=False,sigma_n=0.02):
    """
    Uses a gaussian process to fit existing training data and evaluates it at new test points, returning a vector of means and a covariance matrix.

    Parameters
    ----------
    X_train : (num_train, num_dim) numpy array
        Training data points coordinates.
    y_train : (num_train, 1) numpy array
        Value of the function at the training data points.
    X_test : (num_test, num_dim) numpy array
        Test data points coordinates.
    kernel : function, optional (default None)
        Kernel function that takes two coordinate matrices as input and returns a vector of values; e.g., k = kernel(X1,X2). If `grad_y_train` is not None, this function should also accept a `derivatives=(i,j)` argument, where `i` and `j` are integers between -1 and `num_dim`-1, and return the partial derivative of the kernel with respect to the (first) the `i`-th and (second) `j`-th coordinates of the training data points (-1 denoting no derivative, see `squared_exponential_kernel`). If None, the squared exponential kernel is used.
    X_induced : (num_induced, num_dim) numpy array, optional (default None)
        Inducing points coordinates (see e.g., https://ludwigwinkler.github.io/blog/InducingPoints/). If None, the training data points are used as inducing points.
    grad_y_train : (num_train, num_dim) numpy array, optional (default None)
        Observed gradient of the function at the training data points. If None, the gradient is not used.
    verbose : bool, optional (default False)
        If True, prints information about the training.
    sigma_n : float, optional (default 0.02)
        Noise standard deviation.
    
    Returns
    -------
    mu : (num_test,) numpy array
        Mean of the gaussian process at the test data points.
    sigma : (num_test, num_test) numpy array
        Covariance matrix of the gaussian process at the test data points.

    See also
    --------
    squared_exponential_kernel

    Notes
    -----
    This function is a wrapper for the `gaussian_process_precompute` class, which stores all necessary information to later evaluate the gaussian process at any given test points.  This is useful if the same training data is used to evaluate the gaussian process at multiple test points, as it avoids recomputing the same quantities multiple times.

    Examples
    --------
    TODO
    """
    return gaussian_process_precompute(X_train,y_train,X_induced=X_induced,grad_y_train=grad_y_train,kernel=kernel,verbose=verbose,sigma_n=sigma_n).predict(X_test)


class gaussian_process_precompute:
    def __init__(self,X_train,y_train,X_induced=None,grad_y_train=None,kernel=None,verbose=False,sigma_n=0.02):
        """
        Fits a gaussian process to existing training data, storing all necessary information to later evaluate it at any given test points.

        Parameters
        ----------
        X_train : (num_train, num_dim) numpy array
            Training data points coordinates.
        y_train : (num_train, 1) numpy array
            Value of the function at the training data points.
        kernel : function, optional (default None)
            Kernel function that takes two coordinate matrices as input and returns a vector of values; e.g., k = kernel(X1,X2). If `grad_y_train` is not None, this function should also accept a `derivatives=(i,j)` argument, where `i` and `j` are integers between -1 and `num_dim`-1, and return the partial derivative of the kernel with respect to the (first) the `i`-th and (second) `j`-th coordinates of the training data points (-1 denoting no derivative). If None, the squared exponential kernel is used.
        X_induced : (num_induced, num_dim) numpy array, optional (default None)
            Inducing points coordinates (see e.g., https://ludwigwinkler.github.io/blog/InducingPoints/). If None, the training data points are used as inducing points.
        grad_y_train : (num_train, num_dim) numpy array, optional (default None)
            Observed gradient of the function at the training data points. If None, the gradient is not used.
        verbose : bool, optional (default False)
            If True, prints information about the training.
        sigma_n : float, optional (default 0.02)
            Noise standard deviation.

        Returns
        -------
        precomputed_gaussian_process : instance of class gaussian_process_precompute
            Object that stores all necessary information to later evaluate the gaussian process at any given test points.

        """
        if verbose:
            import time
            t_train_0 = time.time()
        # Store parameteers
       
        self.verbose = verbose
        self.sigma_n = sigma_n
        
        self.X_train = X_train
        self.y_train = y_train
        assert(X_train.shape[0]==y_train.shape[0])
        self.use_gradients = False
        if (grad_y_train is not None):
            self.use_gradients = True
            self.grad_y_train = grad_y_train.flatten('F')
            self.y_train = np.concatenate((self.y_train,self.grad_y_train))
        self.num_train = y_train.shape[0]
        # This part is independent of test data

        # Default kernel
        if (kernel is None):
            self.kernel = squared_exponential_kernel
        else:
            self.kernel = kernel

        # We prefactorize everything
        if (X_induced is not None):

            if self.verbose:
                print("--------- Training Gaussian Process with", X_train.shape[0],"data points and",X_induced.shape[0],"induced points. ---------")
            self.inducing_points = True
            assert(self.sigma_n>0)
            self.X_induced = X_induced
            self.Kmm = cov_matrix_from_function(self.kernel,X_induced,X_induced,use_gradients=self.use_gradients)
            Kmn = cov_matrix_from_function(self.kernel,X_induced,X_train,use_gradients=self.use_gradients)
            self.SIGMA_INV = self.Kmm + ((1/(self.sigma_n**2.0))*Kmn*Kmn.T)
           
            self.sigma_LU = splu(csc_matrix(self.SIGMA_INV))
            # print((Kmn*y_train).shape)
            
            self.mu_m = (1/(self.sigma_n**2.0))*self.Kmm*cg(self.SIGMA_INV,Kmn*self.y_train)[0]
            self.LU = splu(csc_matrix(self.Kmm))
            
            
            self.Kmm_inv_mu_m,_ = cg(self.Kmm,self.mu_m)
            # self.A_m = self.Kmm*self.sigma_LU.solve(self.Kmm.toarray())
            
        else:
            if self.verbose:
                print("--------- Training Gaussian Process with", X_train.shape[0],"data points. ---------")
            K3 = cov_matrix_from_function(self.kernel,X_train,X_train,sparse=True,use_gradients=self.use_gradients)           
            self.inducing_points = False
            self.LU = splu(csc_matrix(K3 + (self.sigma_n**2.0)*eye(K3.shape[0])))
            # plt.spy(K3)
            # plt.show()
            # self.K3_inv_y = self.LU.solve(self.y_train)
            self.K3_inv_y = cg(K3 + (self.sigma_n**2.0)*eye(K3.shape[0]),self.y_train)[0]
            # debug
            # self.K3_inv_y = self.LU.solve(self.y_train)
        self.is_trained = True

        # Timing
        
        if self.verbose:
            t_train_1 = time.time()
            t_train = t_train_1 - t_train_0
            print("Training time:",t_train,"seconds.")

    def predict(self,X_test):
        """
        Evaluates a precomputed gaussian process at the points X_test.
        
        Parameters
        ----------
        X_test : (num_test, dim) numpy array
            The points at which to evaluate the gaussian process.

        Returns
        -------
        mean : (num_test,) numpy array
            The mean of the gaussian process at the points X_test.
        var : (num_test,num_test) numpy array
            The covariance matrix of the gaussian process at the points X_test.
        """
        
        self.num_test = X_test.shape[0]
        if self.verbose:
            import time
            t_test_0 = time.time()
            print("Building K1 matrix...")
            t0 = time.time()
        K1 = cov_matrix_from_function(self.kernel,X_test,X_test,sparse=True,use_gradients=self.use_gradients)
        if self.verbose:
            t1 = time.time()
            print("...built K1 matrix in",t1-t0,"seconds.")
                
        if (self.inducing_points):
            K2 = cov_matrix_from_function(self.kernel,self.X_induced,X_test,use_gradients=self.use_gradients)
            mean = K2.T @ self.Kmm_inv_mu_m
            lu_solve_K2 = self.LU.solve(K2.toarray())
            # cov = K1 - K2.T @ lu_solve_K2 + lu_solve_K2.T @ self.A_m @ lu_solve_K2
            cov = K1 - K2.T @ lu_solve_K2 + lu_solve_K2.T @ self.Kmm @ self.sigma_LU.solve(self.Kmm @ lu_solve_K2)

        else:           
            if self.verbose:
                print("Building K2 matrix...")
                t0 = time.time()
            K2 = cov_matrix_from_function(self.kernel,self.X_train,X_test,use_gradients=self.use_gradients)
            if self.verbose:
                t1 = time.time()
                print("...built K2 matrix in",t1-t0,"seconds.")
            if self.verbose:
                print("Computing mean...")
                t0 = time.time()
            mean = K2.T @ self.K3_inv_y
            if self.verbose:
                t1 = time.time()
                print("...computed mean in",t1-t0,"seconds.")
            if self.verbose:
                print("Computing covariance...")
                t0 = time.time()
            cov = K1 - K2.T @ self.LU.solve(K2.toarray())
            if self.verbose:
                t1 = time.time()
                print("...computed covariance in",t1-t0,"seconds.")

        if self.verbose:
            t_test_1 = time.time()
            t_test = t_test_1 - t_test_0
            print("Total test time:",t_test,"seconds.")
        return mean, cov






def cov_matrix_from_function(ker,X1,X2,use_gradients=False,sparse=True):
    # This computes the covariance matrix between two sets of points. It's a bit more complicated if it needs to account for observed gradients.
    if use_gradients:
        dim = X1.shape[1]
        big_mats = []
        for i in range(dim+1):
            mats = []
            for j in range(dim+1):
                # this should be the derivative of ker wrt the i-1 dimension of x1 and then the j-1 dimension of x2 (if -1, no derivative)
                def fun(x1,x2):
                    return ker(x1,x2,derivatives=(i-1,j-1))
                # fun = ker.partial_derivative(i-1,j-1)
                mats.append(matrix_from_function(fun,X1,X2,sparse=sparse))
            big_mats.append(hstack(mats))
        return vstack(big_mats)
    else:        
        return matrix_from_function(ker,X1,X2,sparse=sparse)




