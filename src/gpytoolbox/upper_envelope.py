import numpy as np

def upper_envelope(V,T,D):
    """Slices tet mesh according per-vertex material soft labels

    Parameters
    ----------
    V : numpy double array
        Matrix of tet mesh vertex coordinates
    T : numpy int array
        Matrix of tet mesh indices into V
    D : numpy double array
        Matrix of per-vertex labels: each column corresponds to a different material and has per-vertex soft labels between zero and one

    Returns
    -------
    U : numpy double array
        Matrix of sliced, output tet mesh vertex coordinates
    G : numpy int array
        Matrix of sliced, output tet mesh indices into U
    LT : numpy int array
        Matrix of hard per-tet labels: in the same style of D, but only taking values one and zero.

    Notes
    -----
    This uses the algorithm described by Abdrashitov et al. in "Interactive Modelling of Volumetric Musculoskeletal Anatomy". We thank the first author, Rinat Abdrashitov, for providing an initial version of the code.

    Examples
    --------
    TODO
    """

    try:
        from gpytoolbox_bindings import _upper_envelope_cpp_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")

    ut, gt, lt = _upper_envelope_cpp_impl(V,T.astype(np.int32),D)

    return ut, gt, lt
