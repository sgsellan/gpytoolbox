# A *Python* Geometry Processing Toolbox

This repo is a work in progress and contains general utility functions I have
needed to code while trying to work on geometry processing research in python.
Most of them will be one-to-one correspondences with
[gptoolbox](https://github.com/alecjacobson/gptoolbox) functions that I have
used in my previous Matlab life and for which I have found no equivalence in
existing libraries. If you find yourself in need of new functionality that is
not in this library, I encourage you to contribute by submitting a pull request
(see below).

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
  documentation standards. Then, validate that all the checks in
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
- Implement 3D version of `regular_square_mesh.py`
- Implement 3D version of `fd_grad.py`
- Implement 3D version of `fd_interpolate.py`
- Implement 3D version of `fd_partial_derivative.py`
- Write clear screened vs un-screened PSR test