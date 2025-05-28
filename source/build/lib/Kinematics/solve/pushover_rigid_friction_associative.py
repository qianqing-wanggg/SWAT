from ..render import MatplotlibRenderer, VtkRenderer, show_brick_with_json_file
from .rigid_friction_associative import solve_infinitefc_associative
from ..calc import cal_gap_2d, cal_gap_3d
import numpy as np
import tqdm
import math
from ..utils.parameter import get_dimension, get_max_step, get_max_iteration, get_beta, get_alpha, get_crack_tolerance, get_convergence_tolerance
from .contface import ContFace
from .finitefc_nonassociative import solve_finitefc_nonassociative
from .util import _update_contp_crack_2d, _update_contp_force_2d, rotate_3d


def solve_pushover_rigid_friction_associative_2d(elems, contps, node_control, disp_incre, max_iteration):
    """Solve the pushover curve from rigid model using associative flow rule

    :param elems: elements of the model
    :type elems: dictionary
    :param contps: contact points of the model
    :type contps: dictionary
    :param node_control: id of the displacement control contact point
    :type node_control: int
    :param disp_incre: incremental value of displacement [dx,dy,dtheta]
    :type disp_incre: list
    :param max_iteration: maximal number of iterations
    :type max_iteration: int
    :raises Exception: disp_incre should be given only in one direction
    :return: forces and displacement of the pushover curve
    :rtype: tuple(list,list)
    """
    component_index = np.nonzero(disp_incre)
    if len(component_index) > 1:
        raise Exception("Only one component of displacement can be controled")

    print("Push over starts:")
    i = 0
    forces = []
    displacements = []
    for i in tqdm.tqdm(range(max_iteration)):
        cal_gap_2d(contps)
        solution = solve_infinitefc_associative(elems, contps)
        if solution['limit_force'] <= 0:
            break
        # print(solution['limit_force'])

        # viewer = MatplotlibRenderer(elems, contps)
        # viewer.plot_displaced(factor=1)
        _control_disp_2d(contps, elems, node_control,
                         disp_incre, component_index[0][0])
        forces.append(solution['limit_force'])
        displacements.append(
            contps[node_control].displacement[component_index[0][0]])
        # viewer = MatplotlibRenderer(elems, contps)
        # viewer.plot_displaced(factor=1e4)
        _displace_model_2d(elems, contps)
        # viewer = MatplotlibRenderer(elems, contps)
        # viewer.plot_displaced(factor=1e0)

        #i += 1

    return forces, displacements


def solve_pushover_rigid_friction_associative_3d(elems, contps, node_control, disp_incre, max_iteration, show_plot=False, save_plot=True, filename=None, render_source="data"):

    component_index = np.nonzero(disp_incre)
    if len(component_index) > 1:
        raise Exception("Only one component of displacement can be controled")

    print("Push over starts:")
    i = 0
    forces = []
    displacements = []
    for i in tqdm.tqdm(range(max_iteration)):
        cal_gap_3d(contps)
        solution = solve_infinitefc_associative(elems, contps)
        if solution['limit_force'] <= 0:
            break

        # if save_plot:
        #     viewer = VtkRenderer(elems, contps)
        #     viewer.plot_displaced(
        #         factor=1, filename=filename+f'/associative_{i:04d}')
        # viewer = MatplotlibRenderer(elems, contps)
        # viewer.plot_displaced(factor=1)
        _control_disp(contps, elems, node_control,
                      disp_incre, component_index[0][0])
        # if save_plot:
        #     viewer = VtkRenderer(elems, contps)
        #     viewer.plot_displaced(
        #         factor=1e0, filename=filename+f'/associative_control_{i:04d}')
        forces.append(solution['limit_force'])
        displacements.append(
            contps[node_control].displacement[component_index[0][0]])
        # viewer = MatplotlibRenderer(elems, contps)
        # viewer.plot_displaced(factor=1e4)
        _displace_model_3d(elems, contps)
        if save_plot:
            viewer = VtkRenderer(elems, contps)
            viewer.plot_displaced(
                factor=1e0, filename=filename+f'/associative_displace_{i:04d}')
        # viewer = MatplotlibRenderer(elems, contps)
        # viewer.plot_displaced(factor=1e4)

        #i += 1

    return forces, displacements


def solve_pushover_finitefc_nonassociative_2d(elems, contps, node_control, disp_incre, max_step, auto_control=False, hload_direction=1, current_alpha=0.3, beta=0.6, max_iteration=10, stop_tolerance=0.0001, _alpha_multiplier=0.5, save_state_plot=False, factor=1, filename="pushover_step"):
    """Solve the pushover curve using nonassociative flow rule

    :param elems: elements of the model
    :type elems: dictionary
    :param contps: contact points of the model
    :type contps: dictionary
    :param node_control: id of the displacement control contact point
    :type node_control: int
    :param disp_incre: incremental value of displacement [dx,dy,dtheta]
    :type disp_incre: list
    :param max_iteration: maximal number of iterations
    :type max_iteration: int
    :raises Exception: disp_incre should be given only in one direction
    :return: forces and displacement of the pushover curve
    :rtype: tuple(list,list)
    """
    component_index = np.nonzero(disp_incre)
    if len(component_index) > 1:
        raise Exception("Only one component of displacement can be controled")
    # # initialization
    # _max_itreration = get_max_iteration()
    # _beta = get_beta()
    # _alpha = get_alpha()
    # _alpha_multiplier = 0.5
    # _tolerance = get_convergence_tolerance()
    # store parameters
    for k, value in contps.items():
        value.stored_ft = value.cont_type.ft
        value.stored_cohesion = value.cont_type.cohesion
    node_control_plot = node_control
    component_index_plot = component_index
    # assemble contact faces
    contfs = dict()
    for p in contps.values():
        if p.faceID not in contfs.keys():
            face = ContFace(p.faceID, p.section_h,
                            p.cont_type.fc, p.cont_type.ft)
            contfs[face.id] = face
            contfs[p.faceID].contps.append(p.id)
            contfs[p.faceID].contps.append(p.counterPoint)
        else:
            contfs[p.faceID].contps.append(p.id)
            contfs[p.faceID].contps.append(p.counterPoint)

    print("Push over starts:")
    i = 0
    forces = []
    displacements = []
    for step in tqdm.tqdm(range(max_step)):
        print(f"Step {step}")
        cal_gap_2d(contps)
        solution = solve_finitefc_nonassociative(
            elems, contps, current_alpha=current_alpha, beta=beta, max_iteration=max_iteration, stop_tolerance=stop_tolerance, _alpha_multiplier=_alpha_multiplier)
        if save_state_plot:
            viewer = MatplotlibRenderer(elems, contps)
            viewer.plot_displaced(factor=factor, invert_y=True, save_fig=True,
                                  show_fig=False, filename=filename+f'_{step}_solved')
        if solution['limit_force'] <= 0:
            break

        # displacements.append(i*disp_incre[component_index[0][0]])

        # **************************************************************Max disp
        if auto_control:
            max_x = 0
            max_id = None
            for p in contps.values():
                if abs(p.displacement[0]) > max_x:
                    max_x = abs(p.displacement[0])
                    max_id = p.id
            print(f"Max x displacement {max_x} point ID is {max_id}")
            node_control = max_id
            disp_incre = (max_x*0.02*hload_direction, 0, 0)
            component_index = [[0]]
        _control_disp_2d(contps, elems, node_control,
                         disp_incre, component_index[0][0])
        forces.append(solution['limit_force'])
        displacements.append(
            contps[node_control_plot].displacement[component_index_plot[0][0]])
        _displace_model_2d(elems, contps)
        if save_state_plot:
            viewer = MatplotlibRenderer(elems, contps)
            viewer.plot_displaced(factor=factor, invert_y=True, save_fig=True,
                                  show_fig=False, filename=filename+f'_{step}_displaced')
        _update_contp_crack_2d(contps, elems, contfs,
                               solution['sliding_points'])

        _update_contp_force_2d(contps, solution['contact_forces'])

        i += 1

    return forces, displacements


def _control_disp(contps, elems, node_control, disp_incre, component_index):
    if get_dimension() == 2 or get_dimension() == 3:
        _control_disp_2d(contps, elems, node_control,
                         disp_incre, component_index)


def _control_disp_2d(contps, elems, node_control, disp_incre, component_index):

    # print(component_index)
    print(
        f"Converting {contps[node_control].displacement[component_index]} to {disp_incre[component_index]} for node {node_control} disp {component_index}")
    factor = disp_incre[component_index] / \
        contps[node_control].displacement[component_index]
    print("factor in displacement control", factor)
    for element in elems.values():
        element.displacement = (np.asarray(
            element.displacement)*factor).tolist()

    # ! Need to update contact point information, to record displacement
    for k, value in contps.items():
        elem_disp = np.asarray(elems[value.cand].displacement)
        #print(f"element displacement {elem_disp}")
        elem_center = elems[value.cand].center
        #print(f"element center {elem_center}")
        node_x = value.coor[0]-elem_center[0]
        node_y = value.coor[1]-elem_center[1]
        new_x = node_x * \
            math.cos(elem_disp[2])+node_y * \
            math.sin(elem_disp[2])+elem_disp[0]+elem_center[0]
        new_y = -node_x * \
            math.sin(elem_disp[2])+node_y * \
            math.cos(elem_disp[2])+elem_disp[1]+elem_center[1]
        value.displacement = [new_x-value.coor[0], new_y-value.coor[1]]
    print("Control point displacement", contps[node_control].displacement)


def _displace_model_3d(elems, contps):
    # update vertices information because the next step could fail
    # element vertices
    for key, value in elems.items():
        vertices = np.array(value.vertices)
        center = np.asarray(value.center)
        vertices_res_center = vertices-center
        rot_angles = np.asarray(value.displacement[3:])
        rotated_vertices_res_center = rotate_3d(
            vertices_res_center, rot_angles, order='xyz')
        disp_center = np.asarray(value.displacement[:3])
        new_vertices = rotated_vertices_res_center+disp_center+center
        value.vertices = new_vertices.tolist()

    for k, value in contps.items():
        disp_center = np.asarray(elems[value.cand].displacement[:3])
        center = np.asarray(elems[value.cand].center)
        point_coord = np.asarray(value.coor)
        point_coord_res_center = point_coord-center
        rot_angles = np.asarray(elems[value.cand].displacement[3:])
        rotated_point_coord_res_center = rotate_3d(
            np.expand_dims(point_coord_res_center, axis=0), rot_angles)[0]
        new_point_coord = rotated_point_coord_res_center+disp_center+center
        value.coor = new_point_coord.tolist()
        value.displacement = [0, 0, 0]

    for key, value in elems.items():
        value.center[0] = value.center[0]+value.displacement[0]
        value.center[1] = value.center[1]+value.displacement[1]
        value.center[2] = value.center[2]+value.displacement[2]
        value.displacement = [0, 0, 0, 0, 0, 0]


def _displace_model_2d(elems, contps):
    #! NEED to update vertices information because the next step could fail
    # element center
    for key, value in elems.items():
        for pt in value.vertices:
            node_x = pt[0]-value.center[0]
            node_y = pt[1]-value.center[1]
            pt[0] = node_x * \
                math.cos(value.displacement[2])+node_y * \
                math.sin(value.displacement[2]) + \
                value.displacement[0]+value.center[0]
            pt[1] = -node_x * \
                math.sin(value.displacement[2])+node_y * \
                math.cos(value.displacement[2]) + \
                value.displacement[1]+value.center[1]

    for k, value in contps.items():
        elem_disp = np.asarray(elems[value.cand].displacement)
        #print(f"element displacement {elem_disp}")
        elem_center = elems[value.cand].center
        #print(f"element center {elem_center}")
        node_x = value.coor[0]-elem_center[0]
        node_y = value.coor[1]-elem_center[1]
        value.coor[0] = node_x * \
            math.cos(elem_disp[2])+node_y * \
            math.sin(elem_disp[2])+elem_disp[0]+elem_center[0]
        value.coor[1] = -node_x * \
            math.sin(elem_disp[2])+node_y * \
            math.cos(elem_disp[2])+elem_disp[1]+elem_center[1]
        value.displacement = [0, 0]

    for key, value in elems.items():
        value.center[0] = value.center[0]+value.displacement[0]
        value.center[1] = value.center[1]+value.displacement[1]
        value.displacement = [0, 0, 0]
