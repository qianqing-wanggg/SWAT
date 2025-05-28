import numpy as np


def normalize(v):
    """normalize a vector

    :param v: vector
    :type v: np.array
    :return: normalized vector
    :rtype: np.array
    """
    norm = np.linalg.norm(v, ord=2)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm
