class ContFace():
    """Contact face class
    """

    def __init__(self, id, height, fc, ft=0):
        """
        :param id: ID of the face
        :type id: int
        :param height: Height of the face
        :type height: float
        :param fc: Compressive strength
        :type fc: float
        """
        self.id = id
        self.contps = []
        self.height = height
        self.fc = fc
        self.ft = ft

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False
