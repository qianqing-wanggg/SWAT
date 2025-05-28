
GOUNDID = 99999999
DIM = 3

# Non-associative solver
MAX_STEP = 3
MAX_ITERATION = 20
CRACK_TOLERANCE = 0.001
CONVERGENCE_TOLERANCE = 0.00001
BETA = 0.6
ALPHA = 0.3


def set_ground_id(id=999999999):
    """Set the ground id to a new value.

    :param id: id, defaults to 999999999
    :type id: int
    """
    global GOUNDID
    GOUNDID = id


def get_ground_id():
    """Get the ground id.

    :return: ground id
    :rtype: int
    """
    global GOUNDID
    return GOUNDID


def set_dimension(d):
    global DIM
    DIM = d


def get_dimension():
    global DIM
    return DIM


def set_max_step(step):
    global MAX_STEP
    MAX_STEP = step


def get_max_step():
    global MAX_STEP
    return MAX_STEP


def set_max_iteration(iteration):
    global MAX_ITERATION
    MAX_ITERATION = iteration


def get_max_iteration():
    global MAX_ITERATION
    return MAX_ITERATION


def set_crack_tolerance(tolerance):
    global CRACK_TOLERANCE
    CRACK_TOLERANCE = tolerance


def get_crack_tolerance():
    global CRACK_TOLERANCE
    return CRACK_TOLERANCE


def set_convergence_tolerance(tolerance):
    global CONVERGENCE_TOLERANCE
    CONVERGENCE_TOLERANCE = tolerance


def get_convergence_tolerance():
    global CONVERGENCE_TOLERANCE
    return CONVERGENCE_TOLERANCE


def set_beta(beta):
    global BETA
    BETA = beta


def get_beta():
    global BETA
    return BETA


def set_alpha(alpha):
    global ALPHA
    ALPHA = alpha


def get_alpha():
    global ALPHA
    return ALPHA
