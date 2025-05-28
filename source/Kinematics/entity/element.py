from ..utils.parameter import get_dimension
class Element():
    """Element
    """

    def __init__(self, id, center, mass, vertices, type='None', shape_file=None):
        """Constructor method

        :param id: ID of the element.
        :type id: int
        :param center: Coordinate of the center of the element.
        :type center: list
        :param mass: Mass of the element.
        :type mass: float
        :param vertices: Vertices' coordinates of the element.
        :type vertices: list of list
        :param type: Verbal identification of the element, defaults to 'None'. Special types are 'ground'
        :type type: str, optional
        :param shape_file: Path to the shape file.
        :type shape_file: str, optional
        """
        self.id = id
        self.center = center
        self.mass = mass
        self.vertices = vertices
        self.dl = []
        self.ll = []
        self.type = type
        self.displacement = [0, 0, 0]
        if get_dimension() == 3:
            self.displacement = [0, 0, 0, 0, 0, 0]
        self.shape_file = shape_file

        self.contps = []

    def to_tuple(self):
        """Convert the element to tuple

        :return: Tuple of the element
        :rtype: tuple
        """
        return (self.id, self.center, self.mass, self.dl, self.ll, self.type, self.displacement, self.shape_file, self.contps)

    def assert_legal(self):
        """Assert if the element is ready to be sent to solver

        :raises Exception: The element has no valid id.
        :raises Exception: The element has no mass.
        """
        if self.id == -1:
            raise Exception(f'undefined element {self.id}')

        if self.mass <= 0:
            raise Exception(f'element{self.id} has negative/zero mass')
