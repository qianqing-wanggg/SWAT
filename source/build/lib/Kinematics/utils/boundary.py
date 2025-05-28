from abc import ABC, abstractmethod
import math
import numpy as np
from .parameter import get_ground_id, get_dimension


class Boundary(ABC):
    @abstractmethod
    def apply(self, model):
        pass


class FixPoint(Boundary):
    @classmethod
    def apply(cls, model, min_bound_z):
        # add ground ghost element as antagonist to the point at the bottom of the wall
        if get_dimension() == 3:
            for p in model.contps.values():
                normal_p = np.array([0, 0, -1])
                if math.isclose(p.coor[2], min_bound_z) and np.allclose(np.array(p.normal)*-1, normal_p, rtol=1e-5):
                    p.anta = get_ground_id()
        elif get_dimension() == 2:
            for p in model.contps.values():
                normal_p = np.array([0, -1])
                if math.isclose(p.coor[1], min_bound_z) and np.allclose(np.array(p.normal)*-1, normal_p, rtol=1e-5):
                    p.anta = get_ground_id()
