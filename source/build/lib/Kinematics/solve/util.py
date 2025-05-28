
import numpy as np
import math


def _update_elem_disp_2d(contps, elems, disps):
    """"Update 2d displacement recorded in elements and contact points

    :param contps: Dictionary of contact points. Key is the point id, value is ContPoint.
    :type contps: dict
    :param elems: Dictionary of elements. Key is the element id, value is Element.
    :type elems: dict
    :param disps: Flattend list of displacement of all elements, in the same order as elements in 'elems' and dx, dy ,thetaz
    :type disps: list
    :raises Exception: Length of 'disps' is 3 times the number of elements.
    """
    if len(disps) != 3*len(elems.values()):
        raise Exception(
            'Displacement list length does not match number of elements')
    disp_index = 0
    for key, value in elems.items():
        value.displacement = [disps[disp_index*3],
                              disps[disp_index*3+1], disps[disp_index*3+2]]
        disp_index += 1
    for k, value in contps.items():
        # print(f"element number {value.cand}")

        elem_disp = elems[value.cand].displacement
        # print(f"element displacement {elem_disp}")
        elem_center = elems[value.cand].center
        # print(f"element center {elem_center}")
        node_x = value.coor[0]-elem_center[0]
        node_y = value.coor[1]-elem_center[1]
        new_x = node_x * \
            math.cos(elem_disp[2])+node_y * \
            math.sin(elem_disp[2])+elem_disp[0]+elem_center[0]
        new_y = -node_x * \
            math.sin(elem_disp[2])+node_y * \
            math.cos(elem_disp[2])+elem_disp[1]+elem_center[1]
        value.displacement = [new_x-value.coor[0], new_y-value.coor[1]]


def _update_elem_disp_3d(contps, elems, disps):
    if len(disps) != 6*len(elems.values()):
        raise Exception(
            'Displacement list length does not match number of elements')
    disp_index = 0
    for key, value in elems.items():
        value.displacement = [disps[disp_index*6],
                              disps[disp_index*6+1], disps[disp_index *
                                                           6+2], disps[disp_index*6+3],
                              disps[disp_index*6+4], disps[disp_index*6+5]]
        disp_index += 1

    for k, value in contps.items():
        disp_center = np.asarray(elems[value.cand].displacement[:3])
        center = np.asarray(elems[value.cand].center)
        point_coord = np.asarray(value.coor)
        point_coord_res_center = point_coord-center
        rot_angles = 1 * \
            np.asarray(elems[value.cand].displacement[3:])  # !important
        rotated_point_coord_res_center = rotate_3d(
            np.expand_dims(point_coord_res_center, axis=0), rot_angles)[0]
        new_point_coord = rotated_point_coord_res_center+disp_center+center
        value.displacement = (new_point_coord-point_coord).tolist()


def rotate_3d(coords, thetas, order='xyz'):
    """Rotate a point cloud.

    Rotate a point cloud using Euler angles with specified rotation sequence
    around origin. Extrinsic and intrinsic rotations are supported.
    Rotation angle is positive if the rotation is counter-clockwise.

    Parameters
    ----------
    coords : (N, 3) array_like
        Points to be rotated.
    thetas : (3,) array_like
        Euler's rotation angles (in radians).
    order : string, optional
        Axis sequence for Euler angles. Up to 3 characters belonging to the set
        {'x', 'y', 'z'} for intrinsic rotations, or {'X', 'Y', 'Z'} for
        extrinsic rotations [default: 'xyz'].

    Returns
    -------
    (N, 3) ndarray
        Rotated points.

    """
    # Rotation matrix around unit x axis
    #thetas[0] = 0
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(thetas[0]), -math.sin(thetas[0])],
                    [0, math.sin(thetas[0]), math.cos(thetas[0])]
                    ])

    # Rotation matrix around unit y axis
    #thetas[1] = 0
    R_y = np.array([[math.cos(thetas[1]), 0, math.sin(thetas[1])],
                    [0, 1, 0],
                    [-math.sin(thetas[1]), 0, math.cos(thetas[1])]
                    ])

    # Rotation matrix around unit z axis
    #thetas[2] = 0
    R_z = np.array([[math.cos(thetas[2]), -math.sin(thetas[2]), 0],
                    [math.sin(thetas[2]), math.cos(thetas[2]), 0],
                    [0, 0, 1]
                    ])

    # Combined rotation matrix
    if order == 'xyz':
        R = np.dot(R_z, np.dot(R_y, R_x))
    elif order == 'yzx':
        R = np.dot(R_x, np.dot(R_z, R_y))

    # Rotate point cloud
    return np.dot(coords, R)


def _update_contp_force_2d(contps, forces):
    for i, value in enumerate(contps.values()):
        value.normal_force = forces[i*2+1]
        value.tangent_force = forces[i*2]


def _update_contp_crack_2d(contps, elems, contfs, sliding_points):
    nb_crack_points = 0
    # detect crack based on faces\
    # # Reinitialize crack state
    # for k, value in contps.items():
    #     value.cont_type.cohesion = value.stored_cohesion
    #     value.cont_type.ft = value.stored_ft
    #     value.crack_state = False
    # for key, contact_face in contfs.items():
    #     point_locations_np = np.zeros((4, 2))
    #     for i in range(len(contact_face.contps)):
    #         point_locations_np[i] = np.asarray(
    #             contps[contact_face.contps[i]].coor)
    #     face_center = np.mean(point_locations_np, axis=0)
    #     total_moment = 0
    #     tension_force = 0

    #     for i, contact_point_id in enumerate(contact_face.contps):
    #         if i % 2 != 0:
    #             coeff = -1
    #         else:
    #             coeff = 1
    #         force = coeff*contps[contact_point_id].normal_force * \
    #             np.asarray(contps[contact_point_id].normal)
    #         lever_ = -np.asarray(contps[contact_point_id].coor)+face_center
    #         moment = np.cross(force, lever_)  # internal moment direction
    #         total_moment += moment
    #         tension_force -= contps[contact_point_id].normal_force
    #     anker_point = np.asarray(contps[contact_face.contps[0]].coor)
    #     anker_point_element_center = np.asarray(
    #         elems[contps[contact_face.contps[0]].cand].center)
    #     anker_vector = anker_point-anker_point_element_center
    #     for i, contact_point_id in enumerate(contact_face.contps):
    #         contact_point_to_center = np.asarray(
    #             contps[contact_point_id].coor)-anker_point_element_center
    #         if np.array_equal(contact_point_to_center, anker_vector):
    #             continue
    #         elif np.cross(contact_point_to_center, anker_vector) == 0:
    #             continue
    #         else:
    #             sign = np.sign(
    #                 np.cross(contact_point_to_center, anker_vector))

    #         if np.sign(total_moment) == 0:  # same compression/tension on the whole section
    #             tensionpoint_id = [contact_point_id,
    #                                contps[contact_point_id].counterPoint]
    #             compressionpoint_id = [
    #                 p for p in contact_face.contps if p not in tensionpoint_id]
    #         elif sign == np.sign(total_moment):
    #             # tension
    #             tensionpoint_id = [contact_point_id,
    #                                contps[contact_point_id].counterPoint]
    #             compressionpoint_id = [
    #                 p for p in contact_face.contps if p not in tensionpoint_id]
    #         elif sign != np.sign(total_moment):
    #             # compression
    #             compressionpoint_id = [contact_point_id,
    #                                    contps[contact_point_id].counterPoint]
    #             tensionpoint_id = [
    #                 p for p in contact_face.contps if p not in compressionpoint_id]
    #         if (6*abs(total_moment)/(contact_face.height**2))+tension_force/contact_face.height >= contact_face.ft or (6*abs(total_moment)/(contact_face.height**2))+tension_force/contact_face.height <= -contact_face.fc:
    #             for p_id in tensionpoint_id:
    #                 contps[p_id
    #                        ].cont_type.cohesion = contps[p_id].c0
    #                 contps[p_id].cont_type.ft = 0
    #                 if contps[p_id].crack_state == False:
    #                     nb_crack_points += 1
    #                 contps[p_id].crack_state = True

    #         if (-6*abs(total_moment)/(contact_face.height**2))+tension_force/contact_face.height >= contact_face.ft or (-6*abs(total_moment)/(contact_face.height**2))+tension_force/contact_face.height <= -contact_face.fc:
    #             for p_id in compressionpoint_id:
    #                 contps[p_id
    #                        ].cont_type.cohesion = contps[p_id].c0
    #                 contps[p_id].cont_type.ft = 0
    #                 if contps[p_id].crack_state == False:
    #                     nb_crack_points += 1
    #                 contps[p_id].crack_state = True
    #         break
    for point_key in sliding_points:
        contps[point_key].cont_type.cohesion = contps[point_key].c0
        contps[point_key].cont_type.ft = 0
        if contps[point_key].crack_state == False:
            nb_crack_points += 1
        contps[point_key].crack_state = True
    # if (6*total_moment/(contact_face.height**2))+tension_force/contact_face.height > contact_face.ft or (6*total_moment/(contact_face.height**2))+tension_force/contact_face.height < -contact_face.fc:
    #     # the min_moment_edge is cracked by tension
    #     # set to default cohesion
    #     contps[min_moment_id].cont_type.cohesion = value.c0
    #     contps[min_moment_id].cont_type.ft = 0

    #     # set to default cohesion
    #     contps[contps[min_moment_id].counterPoint].cont_type.cohesion = value.c0
    #     contps[contps[min_moment_id].counterPoint].cont_type.ft = 0

    #     nb_crack_points += 2
    # if (-6*total_moment/(contact_face.height**2))+tension_force/contact_face.height > contact_face.ft or (-6*total_moment/(contact_face.height**2))+tension_force/contact_face.height < -contact_face.fc:
    #     # set to default cohesion
    #     contps[max_moment_id].cont_type.cohesion = value.c0
    #     contps[max_moment_id].cont_type.ft = 0

    #     # set to default cohesion
    #     contps[contps[max_moment_id].counterPoint].cont_type.cohesion = value.c0
    #     contps[contps[max_moment_id].counterPoint].cont_type.ft = 0
    # #     nb_crack_points += 2

    # for k, value in contps.items():
    #     value.cont_type.cohesion = value.stored_cohesion
    #     value.cont_type.ft = value.stored_ft
    #     value.crack_state = False
    #     if value.normal_force*value.cont_type.mu+value.cont_type.cohesion < abs(value.tangent_force) or value.normal_force < -0.5*value.cont_type.ft*value.section_h:
    #         value.cont_type.cohesion = value.c0  # set to default cohesion
    #         value.cont_type.ft = 0
    #         value.crack_state = True
    #         nb_crack_points += 1
    print(f"{nb_crack_points} points cracked")
    # # a point is cracked if there's displacement between it and its counterpoint
    # nb_crack_points = 0
    # for k, value in contps.items():
    #     value.cont_type.cohesion = value.stored_cohesion
    #     value.cont_type.ft = value.stored_ft
    #     if np.sum((np.array(value.displacement)-np.array(contps[value.counterPoint].displacement))**2) > _crack_tolerance:
    #         value.cont_type.cohesion = value.c0  # set to default cohesion
    #         value.cont_type.ft = 0
    #         nb_crack_points += 1
    # a point is cracked if there's displacement between it and its counterpoint


def _line1(sigma_c, sigma_t, h_j):
    a = (3*h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (3*h_j ** 2*sigma_t)/8
    return (-a, -c)


def _line2(sigma_c, sigma_t, h_j):
    a = (h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (3*h_j ** 2*(sigma_c - sigma_t))/16 - \
        (h_j ** 2*(sigma_c - 3*sigma_t))/8
    return (-a, -c)


def _line3(sigma_c, sigma_t, h_j):
    a = -(h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (h_j ** 2*(3*sigma_c - sigma_t))/8 - \
        (3*h_j ** 2*(sigma_c - sigma_t))/16
    return (-a, -c)


def _line4(sigma_c, sigma_t, h_j):
    a = -(3*h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (3*h_j ** 2*sigma_c)/8
    return (-a, -c)
