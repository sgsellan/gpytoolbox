import numpy as np


def solid_angle(V, F, O):
    # Assuming V and F are already defined as numpy arrays
    VS = V[F[:, 0], :]
    VD = V[F[:, 1], :]

    # 2D vectors from O to VS and VD
    O2VS = np.expand_dims(O[:, :2], axis=1) - np.expand_dims(VS[:, :2], axis=0)
    O2VD = np.expand_dims(O[:, :2], axis=1) - np.expand_dims(VD[:, :2], axis=0)

    # Commented out normalization as per original Matlab script
    # O2VS = O2VS / np.linalg.norm(O2VS, axis=2, keepdims=True)
    # O2VD = O2VD / np.linalg.norm(O2VD, axis=2, keepdims=True)

    S = -np.arctan2(O2VD[:, :, 0] * O2VS[:, :, 1] - O2VD[:, :, 1] * O2VS[:, :, 0], O2VD[:, :, 0] * O2VS[:, :, 0] + O2VD[:, :, 1] * O2VS[:, :, 1])
    return S