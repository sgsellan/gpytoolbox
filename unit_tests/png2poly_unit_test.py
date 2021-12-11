import matplotlib.pyplot as plt
from context import gpytoolbox

# Test 1: Image from the internet
filename = "unit_tests_data/poly.png"
poly = gpytoolbox.png2poly(filename)
# There should be two contours: one for each transition
assert(len(poly)==2)
plt.plot(poly[0][:,0],poly[0][:,1])
plt.plot(poly[1][:,0],poly[1][:,1])
plt.show(block=False)
plt.pause(20)
plt.close()

# Test 2: Image from Adobe Illustrator
filename = "unit_tests_data/illustrator.png"
poly = gpytoolbox.png2poly(filename)
# There should be four contours: one for each transition in each component
assert(len(poly)==4)
plt.plot(poly[0][:,0],poly[0][:,1])
plt.plot(poly[1][:,0],poly[1][:,1])
plt.plot(poly[2][:,0],poly[2][:,1])
plt.plot(poly[3][:,0],poly[3][:,1])
plt.show(block=False)
plt.pause(20)
plt.close()

print("Unit test passed, all asserts passed")