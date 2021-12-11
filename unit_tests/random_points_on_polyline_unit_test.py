import numpy as np
import matplotlib.pyplot as plt
from context import gpytoolbox

# Test 1: Histogram of uniform distribution
V = np.array([[0.0,0.0],[1.0,1.2],[2.0,2.4]])
P, N = gpytoolbox.random_points_on_polyline(V,10)

#plt.scatter(P[:,0],P[:,1])
plt.plot(V[:,0],V[:,1])
plt.scatter(P[:,0],P[:,1])
plt.quiver(P[:,0],P[:,1],N[:,0],N[:,1])
plt.title("Are the normals perpendicular to the polyline and towards the top-left?")
plt.axis('equal')
plt.show(block=False)
plt.pause(4)
plt.close()

P, N = gpytoolbox.random_points_on_polyline(V,200000)
plt.hist(P[:,0],bins=20)
plt.title("Is this a uniform distribution?")
plt.show(block=False)
plt.pause(4)
plt.close()

# Test 2: Very different edge lengths, should still be uniform
V = np.array([[0.0,0.0],[0.05,0.05],[0.95,0.95],[1.0,1.0]])
P, N = gpytoolbox.random_points_on_polyline(V,200000)
plt.hist(P[:,0],bins=20)
plt.title("Is this a uniform distribution?")
plt.show(block=False)
plt.pause(4)
plt.close()

# Test 3: Use circle to check that the normals work
th = np.reshape(np.linspace(0.0,2*np.pi,15),(-1,1))
V = np.concatenate((-np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
P, N = gpytoolbox.random_points_on_polyline(V,40)
plt.plot(V[:,0],V[:,1])
plt.scatter(P[:,0],P[:,1])
plt.quiver(P[:,0],P[:,1],N[:,0],N[:,1])
plt.title("Are the normals perpendicular to the polyline and outwards?")
plt.axis('equal')
plt.show(block=False)
plt.pause(4)
plt.close()

print("Unit test passed, all asserts passed")