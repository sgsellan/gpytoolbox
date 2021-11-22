import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
from metropolis_hastings import metropolis_hastings

# 1D test
# Sample next point from a normal distribution
def next_sample(x0):
    return np.array([multivariate_normal.rvs(x0,0.01)])

# We want to sample a distribution that is proportional to this weird function
# we don't know how to integrate and normalize
def unnorm_distr(x):
    return np.max((1-np.abs(x[0]),1e-8))

S, F = metropolis_hastings(unnorm_distr,next_sample,np.array([0.1]),100000)
# This should look like an absolute value pyramid function
plot1 = plt.figure(1)
plt.hist(np.squeeze(S),100)
plt.title("Does this look like a pyramid with straight sides?")


# 2D test
# Next sample comes from a normal distribution
def next_sample(x0):
    return multivariate_normal.rvs(x0,np.array([[0.01,0.0],[0.0,0.01]]))

# We want to recover a normal function given a function proportional to its density
def unnorm_distr(x):
    return 100*multivariate_normal.pdf(x,mean=np.array([0.0,0.0]),cov=np.array([[0.01,0.0],[0.0,0.01]]))

S, F = metropolis_hastings(unnorm_distr,next_sample,np.array([0.1,0.1]),100000)
plot2 = plt.figure(2)
plt.hist2d(S[:,0],S[:,1],bins=[50,50])
plt.title("Does this look like a normal distribution centered at the origin?")
plt.show(block=False)

plt.pause(200)
#plt.clf()

plt.close(plot1)
plt.close(plot2)

print("Unit test passed, all asserts passed")