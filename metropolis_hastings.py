import numpy as np
import random

def metropolis_hastings(unnorm_distr, next_sample, x0 , num_samples=100):
    # Uses the Metropolis-Hastings algorithm to generate a sequence of random samples from
    # an unknown distribution p assuming one knows a function which is proportional to its density
    #
    # Inputs:
    #       unnorm_distr function returning the value of the known function
    #           which is proportional to the desired distribution density
    #               Inputs:
    #                       x #dim numpy array of sample being considered
    #               Outputs:
    #                       f float function value
    # 
    #       next_sample function returning a candidate next sample from the current
    #           sample x; for example, a Gaussian centered at x
    #               Inputs:
    #                       x #dim numpy array of previous sample point
    #               Outputs:
    #                       x1 #dim numpy array of next candidate sample
    #
    #       num_sample int number of samples in output (this will be *more* than the total number
    #           of considered samples or evaluations of unnorm_distr)
    #
    #       x0 #dim numpy array of first element in the sequence of samples
    #
    # Outputs:
    #       S #num_samples by #dim numpy array of sequence of samples
    #       F #num_samples vector of f evaluated at each row of S
    
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