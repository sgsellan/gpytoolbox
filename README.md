# A *Python* Geometry Processing Toolbox

This repo is a work in progress and contains general utility functions I have
needed to code while trying to work on geometry processing research in python.
Most of them will be one-to-one correspondences with
[gptoolbox](https://github.com/alecjacobson/gptoolbox) functions that I have
used in my previous Matlab life and for which I have found no equivalence in
existing libraries. If you find yourself in need of new functionality that is
not in this library, I encourage you to contribute by submitting a pull request
(see below).

## Installation

`gpyoolbox` uses C++ bindings for certain functionality. Before you use this library, make sure to run
```bash
mkdir build
cd build
cmake ..
make
```
This step may take a few minutes. Once it has completed successfully, you are free to use all `gpytoolbox` functionality in your personal projects by adding to the python path and importing. For example,
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ext/gpytoolbox')))
from gpytoolbox import regular_square_mesh
v, f = regular_square_mesh(10)
```



## How to contribute?

There are three ways in which you can contribute to this library: by fixing a
bug, by expanding existing functionality or by adding new functionality.

- If you identify a bug in an existing function `gpytoolbox/func.py` that you
  know how to fix, please fork this repository and add a check to
  `unit_tests/func_unit_test.py` that replicates the bug (i.e., a check that the
  current code does not pass). Then, fix the bug in `func.py` and verify that
  both the check that you added *but also all previously existing others* in
  `func_unit_test.py` are passed successfully. Then, commit and submit a pull
  request explaining the bug and the fix. If you identify a bug that you *don't*
  know how to fix, please [submit an
  issue](https://github.com/sgsellan/gpytoolbox/issues) instead.
- If you want to expand the functionality of an existing function
  `gpytoolbox/func.py`, please fork this repository and edit `func.py`
  appropriately. *Make sure that your change maintains the previous default
  behaviour* by running `unit_tests/func_unit_test.py` and verifying that all
  checks pass. Then, add checks to `unit_tests/func_unit_test.py` that
  thoroughly evaluate your new functionality, and validate that they pass as
  well. Then, commit and submit a pull request. If you think the default
  behaviour of a function should be changed, please [submit an
  issue](https://github.com/sgsellan/gpytoolbox/issues) instead.
- If you want to add new functionality that is not covered by any of the files
  in `gpytoolbox/*`, then fork this repository and add two files: a
  `gpytoolbox/new_func.py` file that contains a function definition `def
  new_func(...):` and all its functionality, and a
  `unit_tests/new_func_unit_test.py` file that thoroughly validates that the
  function works as intended. Please refer to existing examples like
  `gpytoolbox/fd_partial_derivative.py` and
  `unit_tests/fd_partial_derivative_unit_test.py` for commenting and
  documentation standards. It may be that you need to load some data (like a
  mesh, or an image) to properly test your new function. In that case, add all
  necessary data files to `unit_tests/unit_tests_data/`. Finally, add a line saying `from .new_func import new_func` to `gpytoolbox/__init__.py`. Then, validate that all the checks in
  `new_func_unit_test.py` are passed and add, commit and submit a pull request.
  If you want new functionality to be added but you don't want or know how to
  add it yourself, please [submit an
  issue](https://github.com/sgsellan/gpytoolbox/issues) instead.

If you contribute to this repo in any of the above listed ways, you will be
properly credited both in this page and in the individual files.

## To Do

- Clean up imports
- Implement 3D version of `linear_elasticity_stiffness.py`
- Implement 3D version of `linear_elasticity.py`
- Proper mesh boolean unit test
- Fix upper envelope to not flip tets
- Proper lazy cage unit test
- Write dependencies