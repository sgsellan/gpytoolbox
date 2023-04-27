import numpy as np
from .solid_angle import solid_angle

def winding_number(V, F, O):
    S = solid_angle(V, F, O)
    W = np.sum(S, axis=1) / (2 * np.pi)
    return W