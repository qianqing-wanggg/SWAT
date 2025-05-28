class ContType():
    """Contact type class
    """

    def __init__(self, type, parameters):
        """_summary_

        :param type: Veribal description of the contact type.
        :type type: str
        :param parameters: Parameters of the contact type.
        :type parameters: dict
        :raises Exception: Unknown contact type. Acceptable material types are 'friction', 'friction_cohesion', 'friction_fc_cohesion'.
        """
        self.type = type
        self.mu = -1
        self.cohesion = -1
        self.fc = -1
        self.ft = -1
        self.E = None
        self.lamda = 0.2
        self.Gc = 0
        self.uc_elastic = 0
        self.uc_ultimate = 0
        self.Gf1 = 0
        self.ut_elastic = 0
        self.ut_ultimate = 0
        self.Gf2 = 0
        self.us_elastic = 0
        self.us_ultimate = 0
        if type == 'friction':
            self.mu = parameters['mu']
        elif type == 'friction_cohesion':
            self.mu = parameters['mu']
            self.cohesion = parameters['cohesion']
        elif type == 'friction_fc':
            self.mu = parameters['mu']
            self.fc = parameters['fc']
        elif type == 'friction_fc_cohesion':
            self.mu = parameters['mu']
            self.cohesion = parameters['cohesion']
            self.fc = parameters['fc']
            self.ft = parameters['ft']
        else:
            raise Exception('unknown contact type')
        if 'E' in parameters:
            self.E = parameters['E']
