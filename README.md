# A *Python* Geometry Processing Toolbox

![unit
tests](https://github.com/sgsellan/gpytoolbox/actions/workflows/linux_build.yml/badge.svg)
![unit
tests](https://github.com/sgsellan/gpytoolbox/actions/workflows/macos_build.yml/badge.svg)
![unit
tests](https://github.com/sgsellan/gpytoolbox/actions/workflows/windows_build.yml/badge.svg)

[https://gpytoolbox.org](https://gpytoolbox.org/)

![logo](docs/assets/images/logo.png)

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

## Documentation

You can find documentation for all our functions in [our
website](https://gpytoolbox.org/). You can also view the documentation for a
specific function by running `help(function_name)` or `function_name.__doc__`;
for example,
```python
>>> from gpytoolbox import grad
>>> help(grad)
Finite element gradient matrix

Given a triangle mesh or a polyline, computes the finite element gradient matrix assuming piecewise linear hat function basis.

Parameters
----------
V : numpy double array
    Matrix of vertex coordinates
F : numpy int array, optional (default None)
    Matrix of triangle indices

Returns
-------
G : scipy sparse.csr_matrix
    Sparse FEM gradient matrix

See Also
--------
cotangent_laplacian.

Notes
-----

Examples
--------
TO-DO
```

## Contribute

We hope you find our current version of our library useful. At the same time, we
encourage you to *ask not what Gpytoolbox can do for you, but what you can do
for Gpytoolbox*. 

Since Gpytoolbox is a very young library, we want to make it as easy as possible
for others to contribute to it and help it grow. You can contribute by adding a
new function in a new file inside `src/gpytoolbox/`, or by adding to existing
functions, and [submitting a Pull
Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

We also want to make the contribution process as unintimidating as possible. We
will gladly review and edit your code to make sure it acommodates to our
standards and we have set up many tests that will let us know if your
contribution accidentally breaks anything. If there's any functionality that is
not already in this library, is remotely related to geometry processing, and you
have used or used in any of your past projects, we encourage you to submit it
*as-is* in a Pull Request. We will gladly credit you in the individual function
as well as on this home page.

## License

Gpytoolbox's is released under an MIT license ([see details](/LICENSE.MIT)),
except for files in the `gpytoolbox.copyleft` module, which are under a GPL one
([see details](/LICENSE.GPL)). Functions in the copyleft module must be imported
explicitly; this way, if you import only the main Gpytoolbox module
```python
import gpytoolbox
```
or individual functions from it,
```python
from gpytoolbox import regular_square_mesh, regular_cube_mesh
```
you are only bound by the terms of the permissive MIT license. However, if you
import any functionality from `gpytoolbox.copyleft`; e.g.,
```python
from gpytoolbox.copyleft import mesh_boolean
```
you will be bound by the more restrictive GPL license.

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



## Future to-dos
- Add examples to docstrings.
- Implement tet mesh version of `linear_elasticity_stiffness.py`
- Implement tet mesh version of `linear_elasticity.py`
- Improve `poisson_surface_reconstruction` and make it 3D.
- Write proper BVH structure and efficient signed distances
- Switch to pybind11
- Port fracture modes code
- Port swept volume code
- Add tets to `subdivide.py`
- `angle_defect.py` (which is **zero** at boundary vertices!)
- `dihedral_angles.py`
- Intrinsic Delaunay triangulation
- Nearest point on mesh / Hausdorff distance
- Package for conda distribution
- Add notes on every docstring mentioning libigl implementations
