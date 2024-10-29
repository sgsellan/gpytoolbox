# A *Python* Geometry Processing Toolbox

![unit
tests](https://github.com/sgsellan/gpytoolbox/actions/workflows/linux_build.yml/badge.svg)
![unit
tests](https://github.com/sgsellan/gpytoolbox/actions/workflows/macos_build.yml/badge.svg)
![unit
tests](https://github.com/sgsellan/gpytoolbox/actions/workflows/windows_build.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/gpytoolbox?style=flat&color=blue)](https://pypi.org/project/gpytoolbox/)

[https://gpytoolbox.org](https://gpytoolbox.org/)

![logo](https://github.com/sgsellan/gpytoolbox/raw/main/docs/assets/images/logo.png)

*Authors:* [Silvia Sellán](https://www.silviasellan.com), University of Toronto
and [Oded Stein](https://odedstein.com), University of Southern California

This is a very young library of general geometry processing Python research
utility functions that evolves from our personal student codebases. 

## Installation

### Latest stable release (recommended)

You should be able install the latest release of *Gpytoolbox* with pip:
```bash
python -m pip install gpytoolbox
```
A conda installation will be supported in the future.

### From Git

If you want to build Gpytoolbox from a specific git commit; for example, because
you want to develop for Gpytoolbox or because you want some functionality that
is in the `main` branch but hasn't been pushed to any release yet, you should be
able to do so by cloning [Gpytoolbox's github
repo](https://github.com/sgsellan/gpytoolbox) and running
```bash
python -m pip install numpy
python -m pip install .
```

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

Note that the code that you contribute will be licensed under the MIT license.
Everybody will be able to use this code as long as they credit gpytoolbox (and
not you individually).

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

## Attribution

If you use our library in your research paper, please cite us! You can use the
bibtex block below:

```bibtex
@misc{gpytoolbox,
  title = {{gptyoolbox}: A Python Geometry Processing Toolbox},
  author = {Silvia Sell\'{a}n and Oded Stein and others},
  note = {https://gpytoolbox.org/},
  year = {2023}
}
```

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

### Contributors

- We would like to thank [Michael Jäger](https://github.com/EmJay276) for being Gpytoolbox's first external contributor ([PR #45](https://github.com/sgsellan/gpytoolbox/pull/45), [PR #69](https://github.com/sgsellan/gpytoolbox/pull/69)).
- [Towaki Takikawa](https://github.com/tovacinni) ([PR #49](https://github.com/sgsellan/gpytoolbox/pull/49))
- [Otman Benchekroun](https://github.com/otmanon) ([PR #59](https://github.com/sgsellan/gpytoolbox/pull/59))
- [Abhishek Madan](https://github.com/abhimadan) ([PR #103](https://github.com/sgsellan/gpytoolbox/pull/103))
- [Lukas Hermann](https://github.com/lsh) ([PR #135](https://github.com/sgsellan/gpytoolbox/pull/135))
- [Haoyang Wu](https://github.com/H-YWu) ([PR #142](https://github.com/sgsellan/gpytoolbox/pull/142))

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
query = np.aarray([[0.1,0.1]])
I = in_element_aabb(queries,V,F) # This is a C++ binding
``` -->

# Basic Optimistic Roadmap

Here are some things we think would be nice to incorporate to future versions of
gpytoolbox. If there's one you are missing, feel free to submit a PR adding your item to
this bullet list. If you want to contribute to gpytoolbox, a great way to start is by picking any of the items below that does not have an associated PR yet

## To-do

- Iterative closest point for mesh alignment
- Basic FEM (cotangent matrix, mass matrix, linear elasticity) for tetrahedral meshes
- ARAP for deformation and parametrization
- Exact geodesic distances
- Heat (approximate) geodesic distance
- Blue noise in random mesh sampling
- [Intrinsic](https://www.cs.cmu.edu/~kmcrane/Projects/NavigatingIntrinsicTriangulations/paper.pdf) triangulation routines
- [Fracture mode](https://www.silviasellan.com/pdf/papers/fracture-harmonics.pdf) computation
- Pure-python version of in_element_aabb
- Make all grid sizes, resolutions, etc. into tuples not necessarily numpy
  arrays
- Add notes on every docstring mentioning libigl implementations
- Tetrahedral mesh implementation of `subdivide.py`
- Dihedral angle computation
- `regular_square_mesh` and `regular_cube_mesh` should support different resolutions in `x` and `y`
  direction (sensible default when n_y is None, to n_y=n_x)
  