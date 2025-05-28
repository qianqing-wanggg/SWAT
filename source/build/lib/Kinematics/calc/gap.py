import numpy as np


def cal_gap_2d(contps,force_positive = True):
    """Calculate gap in 2D

    :param contps: contact points of the model
    :type contps: dictionary
    """
    for key, value in contps.items():
        value.gap[0] = 0
        value.gap[1] = cal_ptp_dist_normal_project(
            value.coor, contps[value.counterPoint].coor, contps[value.counterPoint].normal,force_positive=force_positive)


def cal_gap_3d(contps,force_positive=True):
    """Calculate gap in 2D

    :param contps: contact points of the model
    :type contps: dictionary
    """
    for key, value in contps.items():
        value.gap[0] = 0
        value.gap[1] = 0
        value.gap[2] = cal_ptp_dist_normal_project(
            value.coor, contps[value.counterPoint].coor, contps[value.counterPoint].normal,force_positive=force_positive)


def cal_ptp_dist_normal_project(point1, point2, normal2,force_positive = True):
    """Calculate the distance from point1 to point2, projected to the normal of point 2

    :param point1: contact point 1
    :type point1: ContPoint
    :param point2: contact point 2
    :type point2: ContPoint
    :param normal2: normal direction of the face contact point 2 belongs to
    :type normal2: list
    :return: projected distance
    :rtype: float
    """
    vector_2_to_1 = np.asarray(point1)-np.asarray(point2)
    reversed_normal2 = -1*np.asarray(normal2)
    gap = np.dot(vector_2_to_1, reversed_normal2)
    if gap >= 0:
        return gap
    else:
        if force_positive:
            return 0
        else:
            return gap
