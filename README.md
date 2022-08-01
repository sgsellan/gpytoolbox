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

--DESCRIPTION--

## Installation & Use

To write after PyPi deployment

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


## How to contribute?

To write

# TO-DO
## Must do before first PyPi release
- Make every function documented with docstrings so we have pretty auto
  documentation.
- Fix argument conventions (None vs empty array)
- Write unit tests for `bad_quad_mesh_from_quadtree`, `decimate`,
  `do_meshes_intersect`, `edge_indeces` (fix spelling), `lazy_cage`,
  `linear_elasticity_stiffness`, `offset_surface`, `signed_distance_polygon`,
  `subdivide_quad`.
- Figure out what to do about `png2poly`, including writing test.
- Figure out what to do about `write_ply`, including `matplotlib` dependency.
- Fix `test_grad.py`, `test_per_face_normal.py`, `test_per_vertex_normal.py`,
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
- Package for conda distribution
