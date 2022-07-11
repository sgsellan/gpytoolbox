# A *Python* Geometry Processing Toolbox

![unit
tests](https://github.com/sgsellan/gpytoolbox/actions/workflows/ci.yml/badge.svg)

*Author:* [Silvia Sellán](https://www.silviasellan.com), University of Toronto

This repo is a work in progress and contains general utility functions I have
needed to code while trying to work on geometry processing research in Python
3+. Some of them will be one-to-one correspondences with
[gptoolbox](https://github.com/alecjacobson/gptoolbox) functions that I have
used in my previous Matlab life and for which I have found no equivalence in
existing libraries. If you find yourself in need of new functionality that is
not in this library, I encourage you to contribute by submitting a pull request
(see below).

## Template

If you want to build a project using `gpytoolbox`, you can use [our
template](https://github.com/sgsellan/python-project-with-gpytoolbox).

## Installation & Use

Most of the functionality in this library is python-only, and it requires no
installation. To use it, simply clone this repository
```bash
git clone --recursive https://github.com/sgsellan/gpytoolbox.git
```
and install all dependencies
```bash
conda install numpy
conda install -c conda-forge igl
conda install -c conda-forge matplotlib 
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
TBD (If you've compiled this on windows, please submit a pull request with
installation instructions!)

This step may take a few minutes. Once it has completed successfully, you are
free to use the c++ `gpytoolbox` functionality like you would use the pure
Python one; e.g.,
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ext/gpytoolbox')))
from gpytoolbox import regular_square_mesh, in_element_aabb
v, f = regular_square_mesh(10) # This is a pure python function
query = np.array([[0.1,0.1]])
I = in_element_aabb(queries,V,F) # This is a C++ binding
```


## How to contribute?

There are three ways in which you can contribute to this library: by fixing a
bug, by expanding existing functionality or by adding new functionality.

- If you identify a bug in an existing function `gpytoolbox/func.py` that you
  know how to fix, please fork this repository and add a check to
  `test/test_func.py` that replicates the bug (i.e., a check that the current
  code does not pass). Then, fix the bug in `func.py` and verify that both the
  check that you added *but also all previously existing others* in
  `test_func.py` are passed successfully by running `python -m unittest
  test/test_func.py`. Then, commit and submit a pull request explaining the bug
  and the fix. If you identify a bug that you *don't* know how to fix, please
  [submit an issue](https://github.com/sgsellan/gpytoolbox/issues) instead.
- If you want to expand the functionality of an existing function
  `gpytoolbox/func.py`, please fork this repository and edit `func.py`
  appropriately. *Make sure that your change maintains the previous default
  behaviour* by running `python -m unittest test/test_func.py` and verifying
  that all checks pass. Then, add checks to `test/test_func.py` that thoroughly
  evaluate your new functionality, and validate that they pass as well. Then,
  commit and submit a pull request. If you think the default behaviour of a
  function should be changed, please [submit an
  issue](https://github.com/sgsellan/gpytoolbox/issues) instead.
- If you want to add new functionality that is not covered by any of the files
  in `gpytoolbox/*`, then fork this repository and add two files: a
  `gpytoolbox/new_func.py` file that contains a function definition `def
  new_func(...):` and all its functionality, and a `test/test_new_func.py` file
  that thoroughly validates that the function works as intended. Please refer to
  existing examples like `gpytoolbox/fd_partial_derivative.py` and
  `test/test_fd_partial_derivative.py` for commenting and documentation
  standards. It may be that you need to load some data (like a mesh, or an
  image) to properly test your new function. In that case, add all necessary
  data files to `test/unit_tests_data/`. Finally, add a line saying `from
  .new_func import new_func` to `gpytoolbox/__init__.py`. Then, validate that
  all the checks in `test/test_new_func.py` are passed by running `python -m
  unittest test/test_new_func.py` and add, commit and submit a pull request. If
  you want new functionality to be added but you don't want or know how to add
  it yourself, please [submit an
  issue](https://github.com/sgsellan/gpytoolbox/issues) instead.

If you contribute to this repo in any of the above listed ways, you will be
properly credited both in this page and in the individual files.

## To Do

- Implement tet mesh version of `linear_elasticity_stiffness.py`
- Implement tet mesh version of `linear_elasticity.py`
- Proper mesh boolean unit test
- Proper png2poly unit test
- Proper lazy cage unit test
- Write dependencies
- Vectorize `write_ply.py`
- Write `read_ply.py`
- Write `remove_duplicate_vertices.py`
- Write `volumes.py`
- Write quadratic solver with fixed points and linear constraints
- Write `decimate.py` functionality
- Explore exactly which part of png2poly's dependencies we need
- Fix argument conventions (None vs empty array)
- Write unit test for `signed_distance_polygon.py`
- Write python-only build test action
- Package as conda package for easy installation
