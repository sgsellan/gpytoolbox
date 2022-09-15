import numpy as np
import random

def metropolis_hastings(unnorm_distr, next_sample, x0 , num_samples=100):
    """Randomly sample according to an unnormalized distribution.

    Given a function which is proportional to a probabilistic density and a strategy for generating candidate points, returns a set of samples which will asymptotically tend to being a sample a random sample of the unknown distribution.

    Parameters
    ----------
    unnorm_distr : func
        Function returning the value of the known function which is proportional to the desired distribution density
    next_sample : func
        Function returning a candidate next sample from the current
    x0 : numpy array
        Initial sample
    num_samples : int
        Number of samples in output (this will be *more* than the total number of considered samples or evaluations of unnorm_distr)

    Returns
    -------
    S : numpy double array
        Matrix sequence of samples
    F : numpy int array
        Vector of f evaluated at each row of S

    Examples
    --------
    ```python
    from gpytoolbox import metropolis_hastings
    from scipy.stats import multivariate_normal
    import matplotlib.pyplot as plt
    # This is usually a normal
    def next_sample(x0):
        return np.array([multivariate_normal.rvs(x0,0.01)])
    # We want to sample a distribution that is proportional to this weird function we don't know how to integrate and normalize
    def unnorm_distr(x):
        return np.max((1-np.abs(x[0]),1e-8))

    S, F = metropolis_hastings(unnorm_distr,next_sample,np.array([0.1]),1000000)
    # This should look like an absolute value pyramid function
    plt.hist(np.squeeze(S),100)
    plt.show()
    ```
    """
    
    S = np.zeros((num_samples,x0.shape[0]))
    F = np.zeros((num_samples))
    
    S[0,:] = x0 # Hopefully this works!
    
    f0 = unnorm_distr(x0)
    sample_num = 2
    while sample_num<(num_samples+1):
        # Compute next candidate sample
        x1 = next_sample(x0)
        #print(x1)
        # Compute the unnormalized distribution at candidate
        f1 = unnorm_distr(x1)
        #print(f1)
        # Generate random value between 0 and 1
        r = random.uniform(0, 1)
        #print(f1/f0)
        if r<(f1/f0):
            # If accepted, update current sample with candidate sample
            # and add to S and F
            x0 = x1
            f0 = f1
            S[sample_num-1,:] = x1
            F[sample_num-1] = f1
            sample_num = sample_num + 1
    return S,F