import numpy as np
from ..utils.parameter import get_dimension


class ContPoint():
    """Contact point in 3D
    """

    def __init__(self, id, coor, cand, anta, t1, t2, n, cont_type, section_h=None, lever=None, faceID=-1):
        """Constructor method

        :param id: ID of the point
        :type id: int
        :param coor: Coordinates of the point, in the form of x,y,z
        :type coor: list
        :param cand: Candidate element id
        :type cand: int
        :param anta: Antagonist element id
        :type anta: int
        :param t1: The first tangent direction of the point. It should be normalized.
        :type t1: list
        :param t2: The second tangent direction of the point in 3D. It should be normalized.
        :type t2: list
        :param n: Normal direction of the point. It should be normalized.
        :type n: list
        :param cont_type: Contact type of the point
        :type cont_type: ContType
        :param section_h: Section(contact face) height of the face that the point is on
        :type section_h: float
        :param lever: Lever of the point to the center of contact face
        :type lever: float
        :param faceID: ID of the face that the point is on
        :type faceID: int
        """
        self.id = id
        self.coor = coor
        self.cand = cand
        self.anta = anta
        self.normal = n
        self.tangent1 = t1
        if get_dimension() == 3:
            self.tangent2 = t2
        self.cont_type = cont_type

        # for limited compression strength
        if cont_type == 'friction_fc':
            # give erros message if section_h or lever is not defined
            if section_h is None:
                raise Exception(
                    'section_h is not defined for friction_fc contact point')
            if lever is None:
                raise Exception(
                    'lever is not defined for friction_fc contact point')
            if lever < 0:
                raise Exception(
                    f'lever is negative for friction_fc contact point ({self})')

        self.section_h = section_h
        self.lever = lever
        self.faceID = faceID
        self.counterPoint = None
        self.thickness = None

        # for non-associative solution
        self.normal_force = 0
        self.tangent_force = 0
        if get_dimension() == 3:
            self.tangent_force2 = 0
        self.c0 = None
        self.stored_ft = None
        self.stored_cohesion = None
        self.stored_fc = None
        self.crack_state = False
        self.sliding_failure = False
        self.strength_failure = False
        self.opening_failure = False
        self.crushing_failure = False
        self.Dc = 0
        self.Dt = 0
        self.Ds = 0
        # record result
        self.displacement = [0, 0]
        if get_dimension() == 3:
            self.displacement.append(0)

        # for gap

        self.gap = [0, 0]
        if get_dimension() == 3:
            self.gap = [0, 0, 0]

    def to_tuple(self):
        """Convert the contact point to a tuple

        :return: A tuple of the contact point
        :rtype: tuple
        """
        if get_dimension() == 3:
            return (self.id, self.coor, self.cand, self.anta, self.tangent1, self.tangent2, self.normal, self.cont_type, self.section_h, self.lever, self.faceID,
                    self.counterPoint, self.normal_force, self.tangent_force, self.cont_type.mu, self.cont_type.cohesion, self.cont_type.ft, self.cont_type.fc, self.crack_state, self.sliding_failure, self.strength_failure, self.opening_failure, self.crushing_failure, self.displacement, self.gap)
        elif get_dimension() == 2:
            return (self.id, self.coor, self.cand, self.anta, self.tangent1, self.normal, self.cont_type, self.section_h, self.lever, self.faceID,
                    self.counterPoint, self.normal_force, self.tangent_force, self.cont_type.mu, self.cont_type.cohesion, self.cont_type.ft, self.cont_type.fc, self.crack_state, self.sliding_failure, self.strength_failure, self.opening_failure, self.crushing_failure, self.displacement, self.gap)

    def __eq__(self, other):
        if self.coor == other.coor:
            if self.normal == other.normal and self.tangent1 == other.tangent1:
                if get_dimension() == 3:
                    if self.tangent2 == other.tangent2:
                        return True
                elif get_dimension() == 2:
                    return True
        return False

    def __str__(self):
        return f"Contact point information: \nID: {self.id}\nCandidate id: {self.cand}\nAntagonist id: {self.anta}"

    def is_contat_pair(self, other):
        """Checks if two contact points are paired

        :param other: Aother contact point.
        :type other: ContPoint
        :return: 'True' is the two contact points are paired.
        :rtype: bool
        """
        # two points are paired if they have the same coordianates and reversed normals
        if np.allclose(np.array(self.coor), np.array(other.coor), rtol=1e-5):
            if np.allclose(np.array(self.normal), np.array(other.normal)*-1, rtol=1e-5):
                return True
        return False

    def assert_legal(self):
        """Assert if the contact point is ready to be sent to solver

        :raises Exception: The point has no valid id.
        :raises Exception: The point has no coordinate info.
        :raises Exception: The point has no valid contact elements id.
        """
        if self.id == -1:
            raise Exception(f'undefined contact point {self.id}')

        if not self.coor:
            raise Exception(f'contact point {self.id} undefined coordinates')

        if self.cand == -1 or self.anta == -1:
            raise Exception(
                f'contact point {self.id} undefined contact elements')
