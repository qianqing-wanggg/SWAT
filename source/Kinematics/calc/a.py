import numpy as np
from tqdm import tqdm

def cal_A_local_2d(element, point, reverse=False):
    t = np.array(point.tangent1)
    n = np.array(point.normal)
    if reverse:
        t = t*(-1)
        n = n*(-1)

    R = -np.array(point.coor) + np.array(element.center)
    Alocal = np.matrix([
        [-1*t[0], -1*n[0]],
        [-1*t[1], -1*n[1]],
        [-1*float(np.cross(R, t)), -1*float(np.cross(R, n))]
    ])
    return Alocal


def cal_A_global_2d(elems, contps, sparse=False):
    if not sparse:
        Aglobal = np.zeros((3*len(elems), 2*len(contps)))
        row = 0
        for element in elems.values():
            col = 0
            for p_c in contps.values():
                if p_c.cand == element.id:
                    Alocal = cal_A_local_2d(element, p_c)
                    Aglobal[row:row+3, col:col+2] = Alocal
                elif p_c.anta == element.id:
                    Alocal = cal_A_local_2d(element, p_c, reverse=True)
                    Aglobal[row:row+3, col:col+2] = Alocal
                col += 2
            row += 3
        return Aglobal
    else:
        columns_index = []
        columns_values = []
        #col = 0

        # find row index for each element
        element_id_to_index = dict()
        row = 0
        for element in elems.values():
            element_id_to_index[element.id] = row
            row += 3

        N_points = len(contps)
        coord_all_points = np.empty((N_points*2, 2))
        center_elements_all_points = np.empty((N_points*2, 2))
        t1_all_points = np.empty((N_points*2, 2))
        n_all_points = np.empty((N_points*2, 2))
        R_all_points  = np.empty((N_points * 2, 2))

        i = 0
        for p_c in tqdm(contps.values(),total=len(contps),desc='Assemble R matrix'):
            cand_element = elems[p_c.cand]
            anta_element = elems[p_c.anta]
            cand_element_center=np.array(cand_element.center)
            anta_element_center=np.array(anta_element.center)
            point_t1 = np.array(p_c.tangent1)
            point_n = np.array(p_c.normal)
            point_coor= np.array(p_c.coor)
            #assemble the matrix
            ## cand element
            coord_all_points[i] = point_coor
            center_elements_all_points[i] = cand_element_center
            t1_all_points[i] = point_t1
            n_all_points[i]  = point_n
            R_all_points[i]  = -point_coor + cand_element_center
            i += 1
            ## anta element
            coord_all_points[i] = point_coor
            center_elements_all_points[i] = anta_element_center
            t1_all_points[i] = -point_t1
            n_all_points[i]  = -point_n
            R_all_points[i]  = -point_coor + anta_element_center
            i += 1
        #batch processing cross product
        cross_R_t1, cross_R_n = batch_cross_2d(R_all_points, t1_all_points, n_all_points)



        i=0
        for p_c in tqdm(contps.values(),total=len(contps),desc='Calculating A matrix'):
            non_zero_row_n_index = []
            non_zero_row_n_values = []
            non_zero_row_t_index = []
            non_zero_row_t_values = []
            #row = 0
            cand_element = elems[p_c.cand]
            anta_element = elems[p_c.anta]
            Alocal1 =assemble_A_local_2d(cross_R_t1[i,:], cross_R_n[i, :], t1_all_points[i, :], n_all_points[i, :])
            i += 1
            Alocal2 =assemble_A_local_2d(cross_R_t1[i,:], cross_R_n[i, :], t1_all_points[i, :], n_all_points[i, :])
            i += 1

            row_cand = element_id_to_index[cand_element.id]
            non_zero_row_n_index.extend([row_cand, row_cand+1, row_cand+2])
            non_zero_row_n_values.extend(
                np.squeeze(np.asarray(Alocal1[:, 0])).tolist())
            non_zero_row_t_index.extend([row_cand, row_cand+1, row_cand+2])
            non_zero_row_t_values.extend(
                np.squeeze(np.asarray(Alocal1[:, 1])).tolist())
            
            
            row_anta = element_id_to_index[anta_element.id]
            
            non_zero_row_n_index.extend([row_anta, row_anta+1, row_anta+2])
            non_zero_row_n_values.extend(
                np.squeeze(np.asarray(Alocal2[:, 0])).tolist())
            non_zero_row_t_index.extend([row_anta, row_anta+1, row_anta+2])
            non_zero_row_t_values.extend(
                np.squeeze(np.asarray(Alocal2[:, 1])).tolist())
        
            
            # for element in elems.values():
            #     if p_c.cand == element.id:
            #         Alocal = cal_A_local_2d(element, p_c)
            #         non_zero_row_n_index.extend([row, row+1, row+2])
            #         non_zero_row_n_values.extend(
            #             np.squeeze(np.asarray(Alocal[:, 0])).tolist())
            #         non_zero_row_t_index.extend([row, row+1, row+2])
            #         non_zero_row_t_values.extend(
            #             np.squeeze(np.asarray(Alocal[:, 1])).tolist())
            #     elif p_c.anta == element.id:
            #         Alocal = cal_A_local_2d(element, p_c, reverse=True)
            #         non_zero_row_n_index.extend([row, row+1, row+2])
            #         non_zero_row_n_values.extend(
            #             np.squeeze(np.asarray(Alocal[:, 0])).tolist())
            #         non_zero_row_t_index.extend([row, row+1, row+2])
            #         non_zero_row_t_values.extend(
            #             np.squeeze(np.asarray(Alocal[:, 1])).tolist())
            #     row += 3
            columns_index.extend([non_zero_row_n_index, non_zero_row_t_index])
            columns_values.extend(
                [non_zero_row_n_values, non_zero_row_t_values])
            #col += 2
        return (columns_index, columns_values)


# def cal_A_local_3d(element, point, reverse):
#     """Calculate the local stiffness A matrix of a point to an element.

#     :param element: Dictionary of elements
#     :type element: dict
#     :param point: Dictionary of contact points
#     :type point: dict
#     :param reverse: 'False' if the point belongs to the element, defaults to False
#     :type reverse: bool
#     :return: local A matrix
#     :rtype: np.matrix
#     """
#     t1 = np.array(point.tangent1)
#     t2 = np.array(point.tangent2)
#     n = np.array(point.normal)
#     if reverse:
#         t1 = t1*(-1)
#         t2 = t2*(-1)
#         n = n*(-1)

#     R = - np.array(point.coor) + np.array(element.center)
#     # print(R)
#     # see Course statique for the 3D vectorized moments computation
#     # https://moodle.epfl.ch/pluginfile.php/2881982/mod_resource/content/3/CIVIL_124_cours_1_annotated.pdf
#     cross_R_t1 = np.cross(R, t1)
#     cross_R_t2 = np.cross(R, t2)
#     cross_R_n = np.cross(R, n)
#     Alocal = np.matrix([
#         [-1*t1[0], -1*t2[0], -1*n[0]],
#         [-1*t1[1], -1*t2[1], -1*n[1]],
#         [-1*t1[2], -1*t2[2], -1*n[2]],
#         [-1*cross_R_t1[0], -1*cross_R_t2
#             [0], -1*cross_R_n[0]],
#         [-1*cross_R_t1[1], -1*cross_R_t2
#             [1], -1*cross_R_n[1]],
#         [-1*cross_R_t1[2], -1*cross_R_t2
#             [2], -1*cross_R_n[2]]
#     ])
#     return Alocal

def cal_A_local_3d(element_center, point_t1,point_t2,point_n,point_coor):
    """Calculate the local stiffness A matrix of a point to an element.

    :param element: Dictionary of elements
    :type element: dict
    :param point: Dictionary of contact points
    :type point: dict
    :param reverse: 'False' if the point belongs to the element, defaults to False
    :type reverse: bool
    :return: local A matrix
    :rtype: np.matrix
    """

    R = - point_coor + element_center
    # print(R)
    # see Course statique for the 3D vectorized moments computation
    # https://moodle.epfl.ch/pluginfile.php/2881982/mod_resource/content/3/CIVIL_124_cours_1_annotated.pdf
    # cross_R_t1 = np.cross(R, point_t1)
    # cross_R_t2 = np.cross(R, point_t2)
    # cross_R_n = np.cross(R, point_n)
    points = np.stack([point_t1, point_t2, point_n], axis=0)
    cross_R = np.cross(R[None, :], points)
    cross_R_t1, cross_R_t2, cross_R_n = cross_R
    Alocal = np.matrix([
        [-1*point_t1[0], -1*point_t2[0], -1*point_n[0]],
        [-1*point_t1[1], -1*point_t2[1], -1*point_n[1]],
        [-1*point_t1[2], -1*point_t2[2], -1*point_n[2]],
        [-1*cross_R_t1[0], -1*cross_R_t2
            [0], -1*cross_R_n[0]],
        [-1*cross_R_t1[1], -1*cross_R_t2
            [1], -1*cross_R_n[1]],
        [-1*cross_R_t1[2], -1*cross_R_t2
            [2], -1*cross_R_n[2]]
    ])
    return Alocal

def assemble_A_local_3d(cross_R_t1, cross_R_t2, cross_R_n, point_t1, point_t2, point_n):
    Alocal = np.matrix([
        [-1*point_t1[0], -1*point_t2[0], -1*point_n[0]],
        [-1*point_t1[1], -1*point_t2[1], -1*point_n[1]],
        [-1*point_t1[2], -1*point_t2[2], -1*point_n[2]],
        [-1*cross_R_t1[0], -1*cross_R_t2
            [0], -1*cross_R_n[0]],
        [-1*cross_R_t1[1], -1*cross_R_t2
            [1], -1*cross_R_n[1]],
        [-1*cross_R_t1[2], -1*cross_R_t2
            [2], -1*cross_R_n[2]]
    ])
    return Alocal

def assemble_A_local_2d(cross_R_t1, cross_R_n, point_t1, point_n):
    Alocal = np.matrix([
            [-1*point_t1[0], -1*point_n[0]],
            [-1*point_t1[1], -1*point_n[1]],
            [-cross_R_t1[0], -cross_R_n[0]]
        ])
    return Alocal

from concurrent.futures import ProcessPoolExecutor
def wrapper_cal_A_local_3d(args):
    # Unpack arguments for clarity
    center, t1, t2, n, coor = args
    return cal_A_local_3d(center, t1, t2, n, coor)

from numba import njit
@njit
def batch_cross_3d(R, T1, T2, N):
    N_pts = R.shape[0]
    cross_t1 = np.empty_like(R)
    cross_t2 = np.empty_like(R)
    cross_n  = np.empty_like(R)
    for i in range(N_pts):
        cross_t1[i] = np.cross(R[i], T1[i])
        cross_t2[i] = np.cross(R[i], T2[i])
        cross_n[i]  = np.cross(R[i], N[i])
    return cross_t1, cross_t2, cross_n
from numba.np.extensions import cross2d
@njit
def batch_cross_2d(R, T1, N):
    N_pts = R.shape[0]
    cross_t1 = np.empty_like(R)
    #cross_t2 = np.empty_like(R)
    cross_n  = np.empty_like(R)
    for i in range(N_pts):
        cross_t1[i] = cross2d(R[i], T1[i])#cross2d result in a 2D vector with two components exactly the same 
        #cross_t2[i] = np.cross(R[i], T2[i])
        cross_n[i]  = cross2d(R[i], N[i])
        #print(cross_t1[i], cross_n[i])
    return cross_t1, cross_n

def cal_A_global_3d(elems, contps, sparse=False):
    if not sparse:
        Aglobal = np.zeros((6*len(elems), 3*len(contps)))
        row = 0
        for element in elems.values():
            col = 0
            for p_c in contps.values():
                if p_c.cand == element.id:
                    Alocal = cal_A_local_3d(element, p_c, reverse=False)
                    Aglobal[row:row+6, col:col+3] = Alocal
                elif p_c.anta == element.id:
                    Alocal = cal_A_local_3d(element, p_c, reverse=True)
                    Aglobal[row:row+6, col:col+3] = Alocal
                col += 3
            row += 6
    else:
        # find row index for each element
        element_id_to_index = dict()
        row = 0
        for element in elems.values():
            element_id_to_index[element.id] = row
            row += 6
        columns_index = []
        columns_values = []
        col = 0
        N_points = len(contps)
        coord_all_points = np.empty((N_points*2, 3))
        center_elements_all_points = np.empty((N_points*2, 3))
        t1_all_points = np.empty((N_points*2, 3))
        t2_all_points = np.empty((N_points*2, 3))
        n_all_points = np.empty((N_points*2, 3))
        R_all_points  = np.empty((N_points * 2, 3))

        i = 0
        for p_c in tqdm(contps.values(),total=len(contps),desc='Assemble R matrix'):
            cand_element = elems[p_c.cand]
            anta_element = elems[p_c.anta]
            cand_element_center=np.array(cand_element.center)
            anta_element_center=np.array(anta_element.center)
            point_t1 = np.array(p_c.tangent1)
            point_t2 = np.array(p_c.tangent2)
            point_n = np.array(p_c.normal)
            point_coor= np.array(p_c.coor)
            #assemble the matrix
            ## cand element
            coord_all_points[i] = point_coor
            center_elements_all_points[i] = cand_element_center
            t1_all_points[i] = point_t1
            t2_all_points[i] = point_t2
            n_all_points[i]  = point_n
            R_all_points[i]  = point_coor - cand_element_center
            i += 1
            ## anta element
            coord_all_points[i] = point_coor
            center_elements_all_points[i] = anta_element_center
            t1_all_points[i] = -point_t1
            t2_all_points[i] = -point_t2
            n_all_points[i]  = -point_n
            R_all_points[i]  = point_coor - anta_element_center
            i += 1
        #batch processing cross product
        # cross_results = np.cross(R_all_points[:, None, :], np.stack([t1_all_points, t2_all_points, n_all_points], axis=1))
        # cross_R_t1, cross_R_t2, cross_R_n = cross_results[:, 0, :], cross_results[:, 1, :], cross_results[:, 2, :]

        cross_R_t1, cross_R_t2, cross_R_n = batch_cross_3d(R_all_points, t1_all_points, t2_all_points, n_all_points)

        i=0
        for p_c in tqdm(contps.values(),total=len(contps),desc='Calculating A matrix'):
            non_zero_row_1_index = []
            non_zero_row_1_values = []
            non_zero_row_2_index = []
            non_zero_row_2_values = []
            non_zero_row_3_index = []
            non_zero_row_3_values = []
            #row = 0
            cand_element = elems[p_c.cand]
            anta_element = elems[p_c.anta]
            Alocal1 =assemble_A_local_3d(cross_R_t1[i,:], cross_R_t2[i, :], cross_R_n[i, :], t1_all_points[i, :], t2_all_points[i, :], n_all_points[i, :])
            i += 1
            Alocal2 =assemble_A_local_3d(cross_R_t1[i,:], cross_R_t2[i, :], cross_R_n[i, :], t1_all_points[i, :], t2_all_points[i, :], n_all_points[i, :])
            i += 1
            
            # cand_element_center=np.array(cand_element.center)
            # anta_element_center=np.array(anta_element.center)
            # point_t1 = np.array(p_c.tangent1)
            # point_t2 = np.array(p_c.tangent2)
            # point_n = np.array(p_c.normal)
            # point_coor= np.array(p_c.coor)
            # # Prepare input arguments for the two tasks
            # task_args = [
            #     (cand_element_center, point_t1, point_t2, point_n, point_coor),
            #     (anta_element_center, point_t1*(-1), point_t2*(-1), point_n*(-1), point_coor)
            # ]
            # # Run them in parallel
            # with ProcessPoolExecutor(max_workers=2) as executor:
            #     results = list(executor.map(wrapper_cal_A_local_3d, task_args))
            # # Unpack results
            # Alocal1, Alocal2 = results
            # Alocal1 = cal_A_local_3d(cand_element_center, point_t1,point_t2,point_n,point_coor)
            # Alocal2 = cal_A_local_3d(anta_element_center, -point_t1,-point_t2,-point_n,point_coor)

            # Alocal1 = cal_A_local_3d(cand_element, p_c,reverse=False)
            # Alocal2 = cal_A_local_3d(anta_element, p_c, reverse=True)
            row_cand = element_id_to_index[cand_element.id]
            non_zero_row_1_index.extend([row_cand, row_cand+1, row_cand+2, row_cand+3, row_cand+4, row_cand+5])
            non_zero_row_1_values.extend(
                np.squeeze(np.asarray(Alocal1[:, 0])).tolist())
            non_zero_row_2_index.extend([row_cand, row_cand+1, row_cand+2, row_cand+3, row_cand+4, row_cand+5])
            non_zero_row_2_values.extend(
                np.squeeze(np.asarray(Alocal1[:, 1])).tolist())
            non_zero_row_3_index.extend([row_cand, row_cand+1, row_cand+2, row_cand+3, row_cand+4, row_cand+5])
            non_zero_row_3_values.extend(
                np.squeeze(np.asarray(Alocal1[:, 2])).tolist())
            
            row_anta = element_id_to_index[anta_element.id]
            
            non_zero_row_1_index.extend([row_anta, row_anta+1, row_anta+2, row_anta+3, row_anta+4, row_anta+5])
            non_zero_row_1_values.extend(
                np.squeeze(np.asarray(Alocal2[:, 0])).tolist())
            non_zero_row_2_index.extend([row_anta, row_anta+1, row_anta+2, row_anta+3, row_anta+4, row_anta+5])
            non_zero_row_2_values.extend(
                np.squeeze(np.asarray(Alocal2[:, 1])).tolist())
            non_zero_row_3_index.extend([row_anta, row_anta+1, row_anta+2, row_anta+3, row_anta+4, row_anta+5])
            non_zero_row_3_values.extend(
                np.squeeze(np.asarray(Alocal2[:, 2])).tolist())
            # for element in elems.values():
            #     if p_c.cand == element.id:
            #         Alocal = cal_A_local_3d(element, p_c,reverse=False)
            #         non_zero_row_n_index.extend([row, row+1, row+2, row+3, row+4, row+5])
            #         non_zero_row_n_values.extend(
            #             np.squeeze(np.asarray(Alocal[:, 0])).tolist())
            #         non_zero_row_t_index.extend([row, row+1, row+2, row+3, row+4, row+5])
            #         non_zero_row_t_values.extend(
            #             np.squeeze(np.asarray(Alocal[:, 1])).tolist())
            #     elif p_c.anta == element.id:
            #         Alocal = cal_A_local_3d(element, p_c, reverse=True)
            #         non_zero_row_n_index.extend([row, row+1, row+2, row+3, row+4, row+5])
            #         non_zero_row_n_values.extend(
            #             np.squeeze(np.asarray(Alocal[:, 0])).tolist())
            #         non_zero_row_t_index.extend([row, row+1, row+2, row+3, row+4, row+5])
            #         non_zero_row_t_values.extend(
            #             np.squeeze(np.asarray(Alocal[:, 1])).tolist())
            #     row += 6
            columns_index.extend([non_zero_row_1_index, non_zero_row_2_index, non_zero_row_3_index])
            columns_values.extend(
                [non_zero_row_1_values, non_zero_row_2_values, non_zero_row_3_values])
            col += 3
        return (columns_index, columns_values)
    return Aglobal


def calc_global_Y(contps):
    Y = np.zeros((2*len(contps), 3*len(contps)))
    # row=0
    # index = 0
    # for i in range(len(contps.values())):
    #     col = 0
    #     for value in contps.values():
    #         yunit = np.array([
    #             [1.0, -1.0, 0.0],
    #             [-value.cont_type.mu, -value.cont_type.mu, -1.0]])
    #
    #         Y[row:row+2, col:col+3] =yunit
    #     col+=3
    #     row +=2

    col = 0
    row = 0
    for i, value in enumerate(contps.values()):
        col = 3 * i
        row = 2 * i


        yunit = np.array([
            [1.0, -1.0, 0.0],
            [-value.cont_type.mu, -value.cont_type.mu, -1.0]])
        Y[row:row+2, col:col+3] = yunit

    # print("ovo je Y is a.py", Y)
    return Y