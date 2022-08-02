# A *Python* Geometry Processing Toolbox

![unit
tests](https://github.com/sgsellan/gpytoolbox/actions/workflows/linux_build.yml/badge.svg)
![unit
tests](https://github.com/sgsellan/gpytoolbox/actions/workflows/macos_build.yml/badge.svg)
![unit
tests](https://github.com/sgsellan/gpytoolbox/actions/workflows/windows_build.yml/badge.svg)

[https://gpytoolbox.org](https://gpytoolbox.org/)

*Authors:* [Silvia Sellán](https://www.silviasellan.com), University of Toronto
and [Oded Stein](https://odedstein.com), Massachusetts Institute of Technology

This is a very young library of general geometry processing Python research
utility functions that evolves from our personal student codebases. 

## Installation

You should be able install the latest stable release of *Gpytoolbox* with pip:
```bash
python -m pip install gpytoolbox
```
A conda installation will be supported in the future

## Contribute

We hope you find our current version of our library useful. At the same time, we
encourage you to *ask not what Gpytoolbox can do for you, but what you can do for
Gpytoolbox*. 

Since Gpytoolbox is a very young library, we want to make it as easy as possible
for others to contribute to it and help it grow. You can contribute by adding a
new function in a new file inside `src/gpytoolbox/`, or by adding to existing
functions, and [submitting a Pull
Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

We also want to make the contribution process as unintimidating as possible.
We will gladly review and edit your code to make sure it acommodates to our
standards and we have set up many tests that will let us know if your
contribution accidentally breaks anything. If there's any functionality that is
not already in this library, is remotely related to geometry processing, and you
have used or used in any of your past projects, we encourage you to submit it
*as-is* in a Pull Request. We will gladly credit you in the individual function
as well as on this home page.



## Acknowledgements

Several people have, knowingly or unknowingly, greatly contributed to this
library. We are thankful to them:

- [Alec Jacobson](https://www.cs.toronto.edu/~jacobson/) is the author of the
  original Matlab [gptoolbox](https://github.com/alecjacobson/gptoolbox) on
  which we inspired ourselves to create this library. Several of our functions
  are line-by-line translations of his Matlab ones. Thanks, Alec!

- [Nicholas Sharp](https://nmwsharp.com), the author of the game-changing
  geometry visualization library [polyscope](https://polyscope.run/py/), was
  extremely helpful in guiding us through setting up and distributing a Python
  package. Thanks, Nick!

<!-- Most of the functionality in this library is python-only, and it requires no
installation. To use it, simply clone this repository
```bash
git clone --recursive https://github.com/sgsellan/gpytoolbox.git
```
and install all dependencies
```bash
conda install numpy
conda install -c conda-forge igl
conda install -c conda-forge matplotlib 
conda install -c conda-forge scipy
conda install -c conda-forge scikit-sparse
python -m pip install --upgrade pip
python -m pip install polyscope
python -m pip install tetgen
python -m pip install scikit-image
```
Then, use the functions in this library by adding `gpytoolbox` to the python
path and importing; for example,
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'path/to/gpytoolbox')))
from gpytoolbox import regular_square_mesh
v, f = regular_square_mesh(10)
```

Only for certain functionality, `gpyoolbox` uses C++ bindings. These must be
installed only if you wish to use this functionality, and how to do this is
platform-dependent.

### MacOS
Navigate to the cloned repository and run
```bash
mkdir build
cd build
cmake ..
make -j2
```

### Ubuntu
Navigate to the cloned repository and run
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install libmpfr-dev libgmp-dev
mkdir build
cd build
cmake ..
make -j2
```

### Windows
Navigate to the cloned repository and run
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build "." --config Release
```

This step may take a few minutes. Once it has completed successfully, you are
free to use the c++ `gpytoolbox` functionality like you would use the pure
Python one; e.g.,
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ext/gpytoolbox')))
from gpytoolbox import regular_square_mesh, in_element_aabb
v, f = regular_square_mesh(10) # This is a pure python function
query = np.array([[0.1,0.1]])
I = in_element_aabb(queries,V,F) # This is a C++ binding
``` -->

# TO-DO
## Must do before first PyPi release
- Make every function documented with docstrings so we have pretty auto
  documentation.
- Copyleft module
- Fix argument conventions (None vs empty array)
- Write unit tests for `signed_distance_polygon`, `subdivide_quad`.
- Figure out what to do about `png2poly`, including writing test.
- Fix `test_grad.py`, `test_per_face_normals.py`, `test_per_vertex_normals.py`,
  `test_quadtree_laplacian.py`, `test_regular_cube_mesh.py`,
  `test_regular_square_mesh.py`.


## Future to-dos
- Implement tet mesh version of `linear_elasticity_stiffness.py`
- Implement tet mesh version of `linear_elasticity.py`
- Write proper BVH structure and efficient signed distances
- Switch to pybind11
- Port fracture modes code
- Add tets to `subdivide.py`
- `angle_defect.py` (which is **zero** at boundary vertices!)
- `dihedral_angles.py`
- Intrinsic Delaunay triangulation
- Nearest point on mesh / Hausdorff distance
- Package for conda distribution
- Add notes on every docstring mentioning libigl implementations
