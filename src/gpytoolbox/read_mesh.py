import os
import numpy as np
import csv


def read_mesh(file,
    fmt=None,
    return_UV=False,
    return_N=False,
    reader=None):
    # Reads a mesh from file.
    #
    # If you have the approproate C++ extensions installed, this will use a fast
    # C++-based reader. If you do not, this will use a slow python reader.
    #
    # Currently only supports triangle meshes.
    # 
    # Input:
    #       file  the path to the mesh
    #       Optional:
    #                fmt  The file format of the mesh to open.
    #                     If None, try to guess the format from the file
    #                     extension.
    #                     Supported formats: obj
    #                return_UV  Try reading texture coordinates, if they are
    #                           present and the file format supports it.
    #                return_N  Try reading normal coordinates, if they are
    #                          present and the file format supports it.
    #                reader  Which reader engine to use. None, 'C++' or 'Python'.
    #                        If None, will use C++ if available, and else Python.
    #  Output:
    #       V  numpy array of mesh vertex positions
    #       F  numpy array of mesh face indices into V
    #       UV  numpy array of texture coordinates (if requested)
    #       Ft  numpy array of mesh face indices into UV (if requested)
    #       N  numpy array of normal coordinates (if requested)
    #       Fn  numpy array of mesh face indices into N (if requested)
    #

    # Detect format if it has not been specified
    if fmt is None:
        _, fmt = os.path.splitext(file)
    fmt = fmt[1:].lower()

    # Call appropriate helper function to read mesh
    if fmt=='obj':
        V,F,UV,Ft,N,Fn = _read_obj(file,return_UV,return_N,reader)
    else:
        assert False, "Mesh format not supported."

    # Arrange variables for output
    if return_UV and return_N:
        return V,F,UV,Ft,N,Fn
    if return_UV:
        return V,F,UV,Ft
    if return_N:
        return V,F,N,Fn
    return V,F


try:
    # Import C++ reader
    from gpytoolbox_bindings import _read_obj_cpp_impl
    _CPP_READER_AVAILABLE = True
except Exception as e:
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


