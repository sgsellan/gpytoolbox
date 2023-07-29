import os
import numpy as np
import csv
from gpytoolbox.per_face_normals import per_face_normals 
from gpytoolbox.remove_duplicate_vertices import remove_duplicate_vertices


def read_mesh(file,
    fmt=None,
    return_UV=False,
    return_N=False,
    return_C=False,
    reader=None,
    merge_stl=True):
    """Reads a mesh from file.
    
    If you have the approproate C++ extensions installed, this will use a fast
    C++-based reader. If you do not, this will use a slow python reader.
    
    Currently only supports triangle meshes.

    Parameters
    -------
    file : string
        the path the mesh will be read from
    fmt : string, optional (default: None)
        The file format of the mesh to open.
        If None, try to guess the format from the file extension.
        Supported formats: obj, stl
    return_UV : bool, optional (default: None)
        Try reading texture coordinates, if they are present and the file
        format supports it. Only supported for OBJ files.
    return_N : bool, optional (default: None)
        Try reading normal coordinates, if they are present and the file format
        supports it. Only supported for OBJ and PLY files.
    return_C : bool, optional (default: None)
        Try reading color RGBA values, if they are present and the file format
        supports it. Only supported for PLY files.
    reader : string, optional (default: None)
        Which reader engine to use. None, 'C++' or 'Python'.
        If None, will use C++ if available, and else Python.
    merge_stl : bool, optional (default: True)
        If True, will merge the disconnected triangle STL file by removing duplicate vertices.

    Returns
    ----------
    V : (n,3) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh (into V)
    UV : (n_uv,2) numpy array, if requested
        vertex list for texture coordinates
    Ft : (m,3) numpy int array, if requested
        face index list for texture coordinates (into UV)
    N : (n_n,3) numpy array, if requested
        vertex list for normal coordinates
    Fn : (m,3) numpy int array, if requested
        face index list for normal coordinates (into N)
    C : (n,4) or (m,4) numpy int array, if requested
        per-vertex or per-face colors

    Examples
    --------
    ```python
    # Read a mesh in OBJ format
    v,f = gpytoolbox.read_mesh('mesh.obj')
    ```

    ```python
    # Read a mesh in STL format
    v,f = gpytoolbox.read_mesh('mesh.stl',merge_stl=False)
    # This mesh will just be a set of disconnected triangles, so functions like boundary_vertices will just return every vertex
    assert len(gpytoolbox.boundary_vertices(f))==v.shape[0]
    # Read a mesh in STL format, and merge the disconnected triangles
    v,f = gpytoolbox.read_mesh('mesh.stl',merge_stl=True)
    # Now the mesh is a single connected mesh, so boundary_vertices will return the correct result
    assert len(gpytoolbox.boundary_vertices(f))<v.shape[0]
    ```
    
    """

    # Detect format if it has not been specified
    if fmt is None:
        _, fmt = os.path.splitext(file)
    fmt = fmt[1:].lower()

    # Call appropriate helper function to read mesh
    if fmt=='obj':
        V,F,UV,Ft,N,Fn = _read_obj(file,return_UV,return_N,reader)
    elif fmt=='stl':
        V,F = _read_stl(file,merge_stl)
    elif fmt=='ply':
        V,F,N,C = _read_ply(file)
        if return_N:
            Fn = None
    else:
        assert False, "Mesh format not supported."

    # Arrange variables for output
    if return_UV and return_N:
        return V,F,UV,Ft,N,Fn
    if return_UV:
        return V,F,UV,Ft
    if (return_N and return_C):
        return V,F,N,Fn,C
    if return_N:
        return V,F,N,Fn
    if return_C:
        return V,F,C
    return V,F


try:
    # Import C++ reader
    from gpytoolbox_bindings import _read_obj_cpp_impl
    # print("Found pybind bindings!")
    _CPP_READER_AVAILABLE = True
except Exception as e:
    # print("Could not find it!!")
    _CPP_READER_AVAILABLE = False

def _read_obj(file,return_UV,return_N,reader):
    # Private helper function for reading an OBJ file.
    # Currently, only triangle meshes are supported.

    # Pick a reader default
    if reader is None:
        reader = "C++" if _CPP_READER_AVAILABLE else "Python"

    # Select appropriate reader
    if reader=="C++":
        err,V,F,UV,Ft,N,Fn = _read_obj_cpp_impl(file,return_UV,return_N)
        if err != 0:
            if err == -5:
                raise Exception(f"The file {file} could not be opened.")
            elif err == -7:
                raise Exception(f"A line in {file} was ill-formed.")
            elif err == -8:
                raise Exception(f"{file} does not seem to be a triangle mesh.")
            else:
                raise Exception(f"Unknown error {err} reading obj file.")
    elif reader=="Python":
        V,F,UV,Ft,N,Fn = _read_obj_python(file,return_UV,return_N)
    else:
        assert False, "Invalid choice of reader."

    return V,F,UV,Ft,N,Fn


def _read_obj_python(file,return_UV,return_N):
    # Private helper function for reading an OBJ file in pure Python.
    # Currently, only triangle meshes are supported.

    V = None
    UV = None
    N = None
    F = None
    Ft = None
    Fn = None

    with open(file, newline='') as handle:
        reader = csv.reader(handle, delimiter=' ')
        for row in reader:
            # Remove empty strings
            row = [s for s in row if s]
            if len(row)==0:
                continue

            # What kind of row is this?
            s = row[0]
            if s=='#':
                # Comment
                None
            elif s=='v':
                d = len(row)-1
                if V is None:
                    V = np.zeros((1,d), dtype=np.float64)
                else:
                    assert d == V.shape[1], "Inconsistent vertex dimensions"
                    V.resize((V.shape[0]+1,V.shape[1]))
                V[-1,:] = row[1:]
            elif s=='vt':
                if return_UV:
                    d = len(row)-1
                    if UV is None:
                        UV = np.zeros((1,d), dtype=np.float64)
                    else:
                        assert d == UV.shape[1], "Inconsistent texture coord dimensions"
                        UV.resize((UV.shape[0]+1,UV.shape[1]))
                    UV[-1,:] = row[1:]
            elif s=='vn':
                if return_N:
                    d = len(row)-1
                    if N is None:
                        N = np.zeros((1,d), dtype=np.float64)
                    else:
                        assert d == N.shape[1], "Inconsistent normal coord dimensions"
                        N.resize((N.shape[0]+1,N.shape[1]))
                    N[-1,:] = row[1:]
            elif s=='f':
                #Special treatment to separate face/texture/normal
                d = len(row)-1
                assert d == 3, "Only triangle meshes supported"
                f_split = [x.split('/') for x in row[1:]]
                if F is None:
                    F = np.zeros((1,d), dtype=np.int64)
                    if return_UV:
                        Ft = np.zeros((1,d), dtype=np.int64)
                    if return_N:
                        Fn = np.zeros((1,d), dtype=np.int64)
                else:
                    F.resize((F.shape[0]+1,F.shape[1]))
                    if return_UV:
                        Ft.resize((Ft.shape[0]+1,F.shape[1]))
                    if return_N:
                        Fn.resize((Fn.shape[0]+1,F.shape[1]))
                F[-1,:] = [x[0] for x in f_split]
                F[-1,:] -= 1
                if return_UV:
                    Ft[-1,:] = [x[1] if len(x)>1 and len(x[1])>0 else 0 for x in f_split]
                    Ft[-1,:] -= 1
                if return_N:
                    Fn[-1,:] = [x[2] if len(x)>2 and len(x[2])>0 else 0 for x in f_split]
                    Fn[-1,:] -= 1

    return V,F,UV,Ft,N,Fn


def _read_stl(file,merge_stl):
    try:
        from gpytoolbox_bindings import _read_stl_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ read_stl binding, and pure python stl reading is not supported.")
    err,V,F = _read_stl_cpp_impl(file)
    if err != 0:
        if err == -1:
            raise Exception(f"The file {file} exceeds the the ASCII line limit.")
        elif err == -2:
            raise Exception(f"The file {file} was opened but could not be parsed.")
        elif err == -3:
            raise Exception(f"The file {file} seems empty.")
        elif err == -4:
            raise Exception(f"The file {file} does not exist.")
        elif err == -5:
            raise Exception(f"Unknown error reading stl file.")
    if merge_stl:
        V, _, _, F = remove_duplicate_vertices(V,faces=F)
    return V,F

def _read_ply(file):
    try:
        from gpytoolbox_bindings import _read_ply_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ read_ply binding, and pure python ply reading is not supported.")
    err,V,F,N,C = _read_ply_cpp_impl(file)
    if err != 0:
        raise Exception(f"The file {file} could not be read.")
    return V,F,N,C