
from tqdm import tqdm


def cal_anta_id(points):
    """Calculate the antagonist id for each candidate point

    :param points: Dictionary of contact points
    :type points: dict
    :return: Updated dictionary of contact points
    :rtype: dict
    """
    for p_cand in tqdm(points.values()):
        if p_cand.counterPoint is not None:
            continue
        for p_anta in points.values():
            if p_cand == p_anta:
                continue
            if p_anta.counterPoint is not None:
                continue
            if p_cand.is_contat_pair(p_anta):
                p_cand.anta = p_anta.cand
                p_cand.counterPoint = p_anta.id
                p_anta.counterPoint = p_cand.id
                p_anta.anta = p_cand.cand
                # break the inner loop when the pair point is found
                break
    return points


def cal_anta_id_table(points, elems, table):
    """Calculate the antagonist id for each candidate point, assign contact types to points with a table of contact types

    :param points: Dictionary of contact points
    :type points: dict
    :param elems: Dictionary of elements
    :type elems: dict
    :param table: Dictionary that store the stable. Key is a tuple of two element types, value is the contact type
    :type table: dict
    :return: Updated dictionary of contact points
    :rtype: dict
    """
    for p_cand in tqdm(points.values()):
        for p_anta in points.values():
            if p_cand == p_anta:
                continue
            if p_cand.is_contat_pair(p_anta):
                p_cand.anta = p_anta.cand
                p_cand.conterPoint = p_anta.id
                p_cand.cont_type = table[elems[p_cand.cand].type,
                                         elems[p_cand.anta].type]
                # break the inner loop when the pair point is found
                break
    return points


def cal_anta_id_uniform(points, m):
    """Calculate the antagonist id for each candidate point, assign a contact type to all points. 

    :param points: Dictionary of contact points
    :type points: dict
    :param m: Contact type to be assigned to all points
    :type m: ContType
    :return: Updated dictionary of contact points
    :rtype: dict
    """
    ct = m
    for p_cand in tqdm(points.values()):
        for p_anta in points.values():
            if p_cand == p_anta:
                continue
            if p_cand.is_contat_pair(p_anta):
                p_cand.anta = p_anta.cand
                p_cand.conterPoint = p_anta.id
                p_cand.cont_type = ct
                # break the inner loop when the pair point is found
                break
    return points


def cal_anta_id_binary(points, elems, m_same_type, m_diff_type):
    """Calculate the antagonist id for each candidate point. Assign a contact type to points that share the same type of element with its pair point. Assign another contact type to other cases.

    :param points: Dictionary of contact points.
    :type points: dict
    :param elems: Dictionary of elements.
    :type elems: dict
    :param m_same_type: The contact type to be assigned to paired points who have the same element type.
    :type m_same_type: ContType
    :param m_diff_type: The contact type to be assigned to paired points who have different element types.
    :type m_diff_type: ContType
    :return: Updated dictionary of contact points.
    :rtype: ContType
    """
    for p_cand in tqdm(points.values()):
        for p_anta in points.values():
            if p_cand == p_anta:
                continue
            if p_cand.is_contat_pair(p_anta):
                p_cand.anta = p_anta.cand
                p_cand.conterPoint = p_anta.id
                # assign contact _type_
                if elems[p_cand.cand].type == elems[p_cand.anta].type:
                    p_cand.cont_type = m_same_type

                else:
                    p_cand.cont_type = m_diff_type

                # break the inner loop when the pair point is found
                break
    return points
