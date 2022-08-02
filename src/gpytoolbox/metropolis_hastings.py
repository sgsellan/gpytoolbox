import numpy as np
import random

def metropolis_hastings(unnorm_distr, next_sample, x0 , num_samples=100):
    """Finds intersection, union or subtraction of two triangle meshes.

    Given two triangle meshes dA and dB, uses exact predicates to compute the intersection, union or subtraction of the two solids A and B, and output its surface dC

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
    TO-DO
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