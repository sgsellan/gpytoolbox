import numpy as np
from gpytoolbox.decimate import decimate

def swept_volume(V,F,transformations=None,rotations=None,translations=None,scales=None,align_rotations_with_velocity=False,eps=0.05,num_seeds=100,num_faces=None,verbose=False):
    """Find region covered by object a long a trajectory

    Computes the swept volume of a triangle mesh along a trajectory, given as translations keyframes which are interpolated as a Catmull-Rom spline, and rotations which are interpolated using quaternion spherical linear interpolation.

    Parameters
    ----------
    V : numpy double array
        Matrix of mesh vertex coordinates
    F : numpy int array
        Matrix of mesh triangle indices into V
    transformations : list of numpy double array, optional (default None)
        List of transformation matrices in homogeneous coordinates (if not None, superseeds all other trajectory inputs)
    rotations : list of numpy double array, optional (default None)
        List of rotation matrices
    translations : list of numpy double array, optional (default None)
        List of translation vectors (must be set if transformations is None)
    scales : list of doubles, optional (default None)
        List of scaling factors (if None, no scaling is performed)
    align_rotations_with_velocity : bool, optional (default False)
        If rotations is None and this option is True, rotations are chosen *roughyl* such that the shape aligns with the velocity vector
    eps : double, optional (default 0.05)
        Voxel edge-length (finer will be slower but provide a finer output)
    num_seeds : int, optional (default 100)
        Number of seeds to initialize swept volume fronts (see "Swept Volumes via Spacetime Numerical Continuation" for more information). Should be set higher only for extremely complicated self-intersecting paths.
    num_faces : int, optional (default None)
        If not None, will decimate output to have this desired number of faces.
    verbose : bool, optional (default False)
        Whether to print runtime and other performance information.


    Returns
    -------
    U : numpy double array
        Matrix of swept volume mesh vertex coordinates
    G : numpy int array
        Matrix of swept volume mesh triangle indices into U

    See Also
    --------
    decimate.

    Notes
    -----
    This follows the implementation described in "Swept Volumes via Spacetime Numerical Continuation" by Silvia Sellán, Noam Aigerman and Alec Jacobson.

    Examples
    --------
    ```python
    from gpytoolbox import read_mesh, swept_volume
    # Read sample mesh
    v, f = read_mesh("test/unit_tests_data/bunny_oded.obj")
    # Translation vectors to make Catmull-Rom spline
    translation_0 = np.array([0,0,0])
    translation_1 = np.array([1,0,-1])
    translation_2 = np.array([2,0,1])
    translation_3 = np.array([3,0,0])
    translations = [translation_0,translation_1,translation_2,translation_3]
    # Call swept volume function
    u,g = swept_volume(v,f,translations=translations,eps=0.05,
    verbose=False,align_rotations_with_velocity=False)
    ```
    """

    try:
        from gpytoolbox_bindings import _swept_volume_impl
    except:
        raise ImportError("Gpytoolbox cannot import its C++ binding.")



    if(translations is not None):
        transformations = []
        num_transformations = len(translations)
        translations = [np.asarray(t, dtype=np.float64) for t in translations]

        velocity_rotations = None
        if (rotations is None and align_rotations_with_velocity):
            velocity_rotations = velocity_alignment_rotations(translations)

        for i in range(num_transformations):
            this_transformation = np.eye(4)
            this_transformation[0:3,3] = translations[i]
            if (rotations is not None):
                this_transformation[0:3,0:3] = rotations[i]
            elif (velocity_rotations is not None):
                this_transformation[0:3,0:3] = velocity_rotations[i]
            if (scales is not None):
                this_transformation[0:3,0:3] = scales[i]*this_transformation[0:3,0:3]
            transformations.append(this_transformation)

    transformations_big_mat = np.vstack(transformations)
    v,f = _swept_volume_impl(V.astype(np.float64),F.astype(np.int32),transformations_big_mat.astype(np.float64),eps,num_seeds,verbose)

    if(num_faces is not None):
        v,f,_,_ = decimate(v,f,num_faces=num_faces)


    return v,f


def velocity_alignment_rotations(translations):
    # Build a rotation for each keyframe that aligns the shape's local x-axis
    # with the (finite-difference) velocity along the trajectory (see issue #136).
    num = len(translations)
    velocities = []
    for i in range(num):
        if i == 0:
            vel = translations[1] - translations[0]
        elif i == num - 1:
            vel = translations[-1] - translations[-2]
        else:
            vel = translations[i + 1] - translations[i - 1]
        velocities.append(vel / np.linalg.norm(vel))

    reference = np.array([1.0, 0.0, 0.0])
    rotations = [rotation_matrix_from_vectors(reference, velocities[0])]
    for i in range(1, num):
        delta = rotation_matrix_from_vectors(velocities[i - 1], velocities[i])
        rotations.append(delta @ rotations[-1])
    return rotations


def rotation_matrix_from_vectors(vec1, vec2):
    # This function is due to Kevin R. on https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s > 1e-8: # vectors are not (anti)parallel
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    elif c > 0:
        return np.eye(3) # identical directions
    else:
        # antiparallel directions: rotate 180 degrees about any axis
        # perpendicular to a (the minimal rotation is undefined here)
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        kmat = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * kmat.dot(kmat)
