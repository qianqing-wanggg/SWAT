import math
from ..calc.a import cal_A_global_3d, cal_A_global_2d,calc_global_Y
import sys
import mosek
import numpy as np
from ..utils.parameter import get_dimension
from .util import _update_elem_disp_2d, _update_elem_disp_3d, rotate_3d
from ..calc.gap import cal_gap_2d

import numpy as np

def split_A_matrix(Aglobal):
    # get the shape of M
    num_rows, num_cols = Aglobal.shape

    # create matrix An by selecting the third column of M and every third column after that
    An_cols = []
    start_idx = 2
    while start_idx < num_cols:
        An_cols.append(Aglobal[:, start_idx])
        start_idx += 3
    An = np.column_stack(An_cols)

    # create matrix At by selecting the first two columns of M and every third column after that, with a skip of one column
    At_cols = []
    start_idx = 0
    while start_idx < num_cols:
        if start_idx + 2 <= num_cols:
            At_cols.append(Aglobal[:, start_idx:start_idx + 2])
        start_idx += 2
        if start_idx < num_cols:
            start_idx += 1
    At = np.concatenate(At_cols, axis=1)

    return An, At



print_detail = False

def solve_rigid_associative_TH_DB(elems, contps):
    """Solve limit force using associative flow rule considering finite compressive strength

        :param elems: Dictionary of elements
        :type elems: dict
        :param contps: Dictionary of contact points
        :type contps: dict
        :return: Solution
        :rtype: dict
        """
    if get_dimension() == 2:
        return solve_rigid_associative_TH_2d_d(elems, contps)
    elif get_dimension() == 3:
        return solve_rigid_associative_TH_3d_d(elems, contps)


def solve_rigid_associative_TH_2d_d(elems, contps):  #Solves for dispplacements

    """Solve time history in 2D using associative flow rule considering infinite compressive strength

        :param elems: Dictionary of elements
        :type elems: dict
        :param contps: Dictionary of contact points
        :type contps: dict
        :return: Solution
        :rtype: dict
        """
    Aglobal = cal_A_global_2d(elems, contps)

    # result container
    result = dict()
    inf = 0.0
    g = cal_gap_2d(contps)
    # OMM = calc_global_omm(elems)
    Y = calc_global_Y(contps)
    #print("Y je", Y)


    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    # Make mosek environment
    with mosek.Env() as env:
        # Create a task object
        with env.Task(0, 0) as task:
            # Attach a log stream printer to the task
            if print_detail:
                task.set_Stream(mosek.streamtype.log, streamprinter)

            # Bound keys and values for constraints -- flow rule
            bkc = []
            blc = []
            buc = []
            #g_index = 0
            for key, value in contps.items():
                bkc.extend([mosek.boundkey.fx,
                            mosek.boundkey.fx])
                            #mosek.boundkey.fx])
                blc.extend([value.gap[0], value.gap[1]])
                buc.extend([value.gap[0], value.gap[1]])

            #
            # # Bound keys and values for constraints -- contact failure condition - y
            # for key, value in contps.items():
            #     if value.cont_type.type == 'friction':
            #         for i in range(3):
            #             bkc.append(mosek.boundkey.up)
            #             blc.append(-inf)
            #             buc.append(0.0)
            #     else:
            #         raise NameError("unknown contact type!")

            # Bound keys for variables
            bkx = []
            blx = []
            bux = []
            #g_index = 0
            for key, value in elems.items():  # za delta x vektor 3b*1
                for i in range(3):
                    if value.type == "ground":
                        bkx.append(mosek.boundkey.fx)
                        blx.append(0.0)
                        bux.append(0.0)
                    else:
                        bkx.append(mosek.boundkey.fr)
                        blx.append(-inf)
                        bux.append(+inf)

            for key, value in contps.items(): #za lambda (lambda je 3*c+1)
                for i in range(3):
                    bkx.append(mosek.boundkey.lo)
                    blx.append(0.0)
                    bux.append(+inf)


            # Objective coefficients - sta ide uz nepoznate u obj func - g0 #TODO: !!! proveri Y matrix
                    # Objective coefficients
            c = []
            for key, value in elems.items():
                #if value.type != "ground":
                print(f"Element {key} with total load {value.totload}")
                c.extend([-value.totload[0], -value.totload[1], -value.totload[2]])

            # for key, value in elems.items():  # Za masu TODO-solved: za delta x vektor - da li treba nule? ODG: nule jer nema linearnog clana deltax
            #     for i in range(3):
            #         c.append(0.0)

            for key, value in contps.items():  #jer nema lambde u obj fji
                for i in range(3):
                    c.append(0.0)



            # Below is the sparse representation of the A
            # matrix stored by column.
            asub = []
            aval = []
            #
            elem_index = 0
            #
            # print("Aglobal", Aglobal)
            start_row_for_extension = 3 * len(elems)
            Aglobal_tr = Aglobal.transpose()
            for i, value in enumerate(elems.values()):
                # if value.type != "ground":
                for j in range(3):  # 2variables(t,n)*1nodes*
                    col = i * 3 + j
                    col_index = []
                    col_value = []
                    for row in range(len(contps) * 2):
                        if Aglobal_tr[row][col] != 0:
                            col_index.append(row)
                            col_value.append(Aglobal_tr[row][col])

                    asub.append(col_index)
                    aval.append(col_value)
            # Start appending new rows from this index
            start_row_index = 3 * len(elems)

            # Number of contact points, assuming len(contps) gives the total number of contact points
            num_contact_points = len(contps)

            # Values to be added in each new row, structured as [value1, value2]
            new_row_values = [[-1,0.7], [1, 0.7], [0, 1]]

            # Iterate through each contact point, adding new rows for each
            for i in range(num_contact_points):
                for j in range(3):  # Each contact point adds 3 new rows
                    # Calculate the current row index to append the new values
                    current_row_index = start_row_index + i * 3 + j

                    # Determine the column indices for the current set of values
                    # This assumes that columns are to be filled in pairs starting from 0,1 then 2,3, etc.
                    col_indices = [i * 2, i * 2 + 1]

                    # Append the column indices and values for the current row
                    col_index = col_indices
                    col_value = new_row_values[j]

                    # Append these to your asub and aval structures
                    asub.append(col_index)
                    aval.append(col_value)
                    # # Here we don't need to adjust for (i * 3 + j) * 3, just start from the calculated start row
                    # extension_start_row = start_row_for_extension + math.floor(
                    #     col / 2) * 3  # Adjusted to add rows vertically under each element
                    #
                    # # Extend the indices and values vertically
                    # extension_values = [[1, -0.7], [-1, -0.7], [0, -1]]
                    # for k in range(3):  # Adding 3 rows of values for each set
                    #     new_row = extension_start_row + k
                    #     col_index.append(new_row)  # Update the column index
                    #     col_value.extend(extension_values[k])  # Extend with the pre-defined values


                   # print(aval)

                    # row = i * 3 + j
                    # _start_col = len(elems)*3 + math.floor(col/2)*3
                    # row_index.extend(list(range(_start_col, _start_col+3)))
                    # if row % 2 == 0:
                    #     col_value.extend([1, -1, 0])
                    # else:
                    #     col_value.extend(
                    #         [-0.7, -0.7, -1])
                    #
                    #     asub.append(col_index)
                    #     aval.append(col_value)


            #         asub.append(col_index)
            #         aval.append(col_value)
            # print("Asub posle Agl",asub)
            # print("Aval posle Agl", aval)
            # for col in range(len(contps)*3):
            #     col_index =[]
            #     col_value =[]
            #     for row in range(len(contps)*2):
            #         if Y[row][col] != 0:
            #             col_index.append(row)
            #             col_value.append(-1*Y[row][col])
            #     asub.append(col_index)
            #     aval.append(col_value)
            #
            # print("Asub posle Y", asub)
            # print("Aval posle Y", aval)
            # for i, value in enumerate(contps.values()):
            #     for j in range(2):  # 2variables(t,n)*1nodes*
            #         col = i*2+j
            #         col_index = []
            #         col_value = []
            #         for row in range(len(elems)*3):
            #             if Aglobal[row][col] != 0:
            #                 col_index.append(row)
            #                 col_value.append(Aglobal[row][col])
            #         _start_row = len(elems)*3 + math.floor(col/2)*3
            #         col_index.extend(
            #             list(range(_start_row, _start_row+3)))
            #         if col % 2 == 0:
            #             col_value.extend([1, -1, 0])
            #         else:
            #             col_value.extend(
            #                 [-value.cont_type.mu, -value.cont_type.mu, -1])
            #
            #         asub.append(col_index)
            #         aval.append(col_value)

            numvar = len(bkx)
            numcon = len(bkc)

            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(numcon)

            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numvar)

            for j in range(numvar):
                # Set the linear term c_j in the objective.

                task.putcj(j, c[j])

                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(j, bkx[j], blx[j], bux[j])

                # Input column j of A
                task.putacol(j,  # Variable (column) index.
                             # Row index of non-zeros in column j.
                             asub[j],
                             aval[j])  # Non-zero Values of column j.

            # Set the bounds on constraints.
             # blc[i] <= constraint_i <= buc[i]

            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])

            # Set up and input quadratic objective
            qsubi = []
            qsubj = []
            qval = []
            mass_matrix_index = 0
            for key, value in elems.items():

                # qsubi.extend([3+mass_matrix_index, 3+mass_matrix_index + 1, 3+mass_matrix_index + 2])
                # qsubj.extend([3+mass_matrix_index, 3+mass_matrix_index + 1, 3+mass_matrix_index + 2])
                qsubi.extend([mass_matrix_index, mass_matrix_index + 1, mass_matrix_index + 2])
                qsubj.extend([mass_matrix_index, mass_matrix_index + 1, mass_matrix_index + 2])

                qval.extend([value.omm[0][0], value.omm[1][1],
                             value.omm[2][2]])
                print("Mass matrix", value.omm)

                mass_matrix_index += 3

            for key, value in contps.items():
                qsubi.extend([mass_matrix_index, mass_matrix_index + 1, mass_matrix_index + 2])
                qsubj.extend([mass_matrix_index, mass_matrix_index + 1, mass_matrix_index + 2])

                qval.extend([0,0,0])
                mass_matrix_index += 3
            task.putqobj(qsubi, qsubj, qval)

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

            # Optimize
            task.writedata("data.opf")
            task.optimize()
            # Print a summary containing information
            # about the solution for debugging purposes

            task.solutionsummary(mosek.streamtype.msg)
            # Get status information about the solution
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)

            # Output a solution
            xx = [0.] * numvar
            y = [0.] * numcon

            task.getxx(mosek.soltype.itr,
                       xx)
            task.gety(mosek.soltype.itr,
            y)
            print("Lagrange multipliers for the constraints:", y)
            if solsta == mosek.solsta.optimal:
                print("Optimal solution: %s" % xx)
            elif solsta == mosek.solsta.dual_infeas_cer:
                print("Primal or dual infeasibility.\n")
            elif solsta == mosek.solsta.prim_infeas_cer:
                print("Primal or dual infeasibility.\n")
            elif mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")

            n = len(elems) * 3
            c = len(contps)*2
            xx= [-item for item in xx]
            displ_x = xx[0:n]
            #displ_x= [-item for item in displ_x]
            result['displacements'] = displ_x
            print("Disp_x", displ_x)
            delta_rot = displ_x[::5]
            delta_x = displ_x[::3]
            y = [-item for item in y]
            print("Duzina y:",len(y))
            contacts = y[-c:]
            print("Duzina contacts", len(contacts))
            #contacts= [-item for item in contacts]
        # # normalize the displacement  #TODO: Check: to changed .ll to .totload?
        sum = 0
        element_index = 0
        for k, value in elems.items():
            sum += value.ll[0] * xx[element_index * 3] + value.ll[1] * \
                   xx[element_index * 3 + 1] + value.ll[2] * xx[element_index * 3 + 2]
            element_index += 1
        if sum == 0:
            #result["displacements"] = displ_x[0:len(elems) * 3]
            result["displacements"] = displ_x
        else:
            result["displacements"] = (
                    np.array(displ_x) / sum).tolist()

        #result['suc'] = suc
        _update_elem_disp_2d(contps, elems, result["displacements"])

        _displace_model(elems, contps)
        _update_elem_disp_2d(contps, elems, result["displacements"])

        #_displace_model(elems, contps)
    #return result, delta_rot, delta_x
    return result['displacements'], delta_rot, contacts ,solsta
    #return result


def _displace_model(elems, contps):
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
       # print(value.coor[0], value.coor[1])
        node_y = value.coor[1]-elem_center[1]
        value.coor[0] = node_x * \
            math.cos(elem_disp[2])+node_y * \
            math.sin(elem_disp[2])+elem_disp[0]+elem_center[0]
        value.coor[1] = -node_x * \
            math.sin(elem_disp[2])+node_y * \
            math.cos(elem_disp[2])+elem_disp[1]+elem_center[1]

    for key, value in elems.items():
        value.center[0] = value.center[0]+value.displacement[0]
        value.center[1] = value.center[1]+value.displacement[1]
        value.displacement = [0, 0, 0]


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
