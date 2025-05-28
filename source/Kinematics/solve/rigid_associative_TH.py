import math
from ..calc.a import cal_A_global_3d, cal_A_global_2d
import sys
import mosek
import numpy as np
from ..utils.parameter import get_dimension
from .util import _update_elem_disp_2d, _update_elem_disp_3d, rotate_3d
from scipy import sparse
#import scs
import tqdm
from ..calc.gap import cal_gap_2d, cal_gap_3d
from decimal import *
#import cvxpy as cp


def split_A_matrix(Aglobal):
    # get the shape of M
    num_rows, num_cols = Aglobal.shape

    # create matrix An by selecting every third column of M starting with the first one
    An = Aglobal[:, ::3]

    # create matrix At by selecting columns of M as described
    At_cols = []
    start_idx = 1
    while start_idx < num_cols:
        if start_idx + 2 <= num_cols:
            At_cols.append(Aglobal[:, start_idx:start_idx + 2])
        start_idx += 3
    At = np.concatenate(At_cols, axis=1)

    return An, At


print_detail = True

def solve_rigid_associative_TH(elems, contps):
    """Solve limit force using associative flow rule considering finite compressive strength

        :param elems: Dictionary of elements
        :type elems: dict
        :param contps: Dictionary of contact points
        :type contps: dict
        :return: Solution
        :rtype: dict
        """
    if get_dimension() == 2:
        return solve_rigid_associative_TH_2d_f(elems, contps)
    # elif get_dimension() == 3:
    #     return solve_rigid_associative_TH_3d_f(elems, contps)


def solve_rigid_associative_TH_2d_f(elems, contps): # solve for forces
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
    # g = cal_gap_2d(contps)
    # OMM = calc_global_omm(elems)
    #
    # Y = calc_global_Y(contps)



    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()
    # contact_forces = []
    # displ_x =  []
    # displ_y = []
    # displ_rot = []
    # limit_force = 0
    # Make mosek environment
    with mosek.Env() as env:
        # Create a task object
        with env.Task(0, 0) as task:
            # Attach a log stream printer to the task
            if print_detail:
                task.set_Stream(mosek.streamtype.log, streamprinter)

            # Bound keys and values for constraints -- force equilibrium
            bkc = []
            blc = []
            buc = []
            for key, value in elems.items():
                if value.type == "ground":
                    bkc.extend([mosek.boundkey.fr,
                                mosek.boundkey.fr,
                                mosek.boundkey.fr])
                    blc.extend([-inf, -inf, -inf])
                    buc.extend([inf, inf, inf])
                else:
                    bkc.extend([mosek.boundkey.fx,
                                mosek.boundkey.fx,
                                mosek.boundkey.fx])
                    print("value.totload", value.totload)
                    blc.extend([value.totload[0], value.totload[1], value.totload[2]])
                    buc.extend([value.totload[0], value.totload[1], value.totload[2]])


            # Bound keys and values for constraints -- contact failure condition
            for key, value in contps.items():
                if value.cont_type.type == 'friction':
                    for i in range(3):
                        bkc.append(mosek.boundkey.up)
                        blc.append(-inf)
                        buc.append(0.0)
                else:
                    raise NameError("unknown contact type!")

            # Bound keys for variables
            bkx = []
            blx = []
            bux = []
            g_index = 0

            # for key, value in elems.items():  # za r vektor
            #     for i in range(3):
            #
            #         bkx.append(mosek.boundkey.fr)
            #         blx.append(-inf)
            #         bux.append(+inf)
            for idx, (key, value) in enumerate(elems.items()):  # za r vektor
                for i in range(3): #DA LI TREBA DA r od Ground bude poz? vrvt ne jer dx moze biti u minusu

                        if value.type == "ground":
                            bkx.append(mosek.boundkey.fx)
                            blx.append(0.0)
                            bux.append(0.0)
                        else:
                            bkx.append(mosek.boundkey.fr)
                            blx.append(-inf)
                            bux.append(+inf)

            for key, value in contps.items(): #za c
                for i in range(2):
                    bkx.append(mosek.boundkey.fr)
                    blx.append(-inf)
                    bux.append(+inf)

                    g_index += 1



            # Objective coefficients - sta ide uz nepoznate u obj func
                    # Objective coefficients

            c = []

            i =0
            for key, value in elems.items():  #za r vektor - NEmas linearni deo uz r
                for i in range(3):
                    c.append(0.0)

            for key, value in contps.items():
                # for i in range(2):  # 2variables(t,n)*1nodes*2contact faces
                c.extend([-value.gap[0], -value.gap[1]])
                # print(-g[g_index])



            #print(c)


            # for key, value in elems.items():  # za r vektor #TODO: Da li treba c 1.0 jer je r uz kvadratni deo, uz M?
            #     for i in range(3): #jer nema lambde u obj fji
            #         c.append(1.0)
            # for key, value in elems.items():  #TODO-solved: za delta x vektor - da li treba nule? ODG: nule jer nema linearnog clana deltax
            #     for i in range(3):
            #         c.append(0.0)


            # Below is the sparse representation of the A
            # matrix stored by column.
            asub = []
            aval = []
            i=0

            # col_index = []
            # col_value = []
            #
            # # sta ide uz r - jedinice
            # i = 0
            # for col in range(len(elems) * 3):
            #
            #     col_index.extend([col])
            #     col_value.append(
            #         1.0)
            #     i += 1
            #
            #     asub.append(col_index)
            #     aval.append(col_value)

 # Temporary list to hold column values for the current vector
            col_index = []
            col_value = []
            i = 0
            for i in range(len(elems)*3):
                col_index = [i ]  # Creating indices that increment by three for each vector
                col_value = [1.0]  # Only one value in each vector, which is one



                # Append the constructed lists to asub and aval
                asub.append(col_index)
                aval.append(col_value)

            # print('Asub posle r', asub)
            # print('Aval posle r', aval)

            for i, value in enumerate(contps.values()):
                for j in range(2):  # 2variables(t,n)*1nodes*
                    col = i * 2 + j
                    col_index = []
                    col_value = []
                    for row in range(len(elems) * 3):
                        if Aglobal[row][col] != 0:
                            col_index.append(row)
                            col_value.append(Aglobal[row][col])
                    _start_row = len(elems) * 3 + math.floor(col / 2) * 3
                    col_index.extend(
                        list(range(_start_row, _start_row + 3)))
                    if col % 2 == 0:
                        col_value.extend([1, -1, 0])
                    else:
                        col_value.extend(
                            [-value.cont_type.mu, -value.cont_type.mu, -1])

                    asub.append(col_index)
                    aval.append(col_value)

            # print('Asub posle A i Y', asub)
            # print('Aval posle A i Y', aval)

            # print(asub)
            # print('aval',aval)
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
            #inverted_mass_matrix = np.linalg.inv(OMM)
            mass_matrix_index = 0
            for key, value in elems.items():


                # qsubi.extend([3+mass_matrix_index, 3+mass_matrix_index + 1, 3+mass_matrix_index + 2])
                # qsubj.extend([3+mass_matrix_index, 3+mass_matrix_index + 1, 3+mass_matrix_index + 2])
                #
                # qval.extend([-value.invomm[0][0], -value.invomm[1][1],
                #              -value.invomm[2][2]])
                qsubi.extend([mass_matrix_index, mass_matrix_index + 1, mass_matrix_index + 2])
                qsubj.extend([mass_matrix_index, mass_matrix_index + 1, mass_matrix_index + 2])

                qval.extend([-value.invomm[0][0], -value.invomm[1][1], -value.invomm[2][2]])
                print("invomm", value.invomm)
                mass_matrix_index += 3
                #mass_matrix_index += 3
            for key, value in contps.items():
                qsubi.extend([mass_matrix_index, mass_matrix_index + 1])
                qsubj.extend([mass_matrix_index, mass_matrix_index + 1])
                qval.extend([0, 0])
                mass_matrix_index += 2
            task.putqobj(qsubi, qsubj, qval)

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)
            #task.putintparam(mosek.iparam.infeas_report_level, 10)
            # task.putintparam(mosek.iparam.optimizer,  # TODO: Ili dparam
            #                  mosek.optimizertype.intpnt)
                             # From QQ:
                             # task.putintparam(mosek.iparam.optimizer, #TODO: Ili dparam
                             #                  mosek.optimizertype.intpnt) #Dual simplex moze samo za lin fje, interior point za quad
                             #
                             # task.putintparam(mosek.iparam.presolve_use,
                             #                  mosek.presolvemode.off)
                             # task.putintparam(mosek.iparam.optimizer,
                             #                  mosek.optimizertype.dual_simplex)
                             # # Optimize
                             # task.putintparam(mosek.iparam.presolve_use,
                             #                  #mosek.presolvemode.off)
            task.writedata("data.opf")
            task.optimize()

            task.solutionsummary(mosek.streamtype.msg)
            # Get status information about the solution
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)

            xx = [0.] * numvar
            xc = [0.] * numcon
            #print("Duzina xx je", len(xx))

            y = [0.] * numcon
            suc = [0.] * numcon
            task.getxx(mosek.soltype.itr,
                       xx)

            task.gety(mosek.soltype.itr, y)
            print("Lagrange multipliers for the constraints:", y)

            if solsta == mosek.solsta.optimal:
                print("Optimal solution: %s" % xx)
            elif solsta == mosek.solsta.dual_infeas_cer:
                print("Dual infeasibility.\n")
            elif solsta == mosek.solsta.prim_infeas_cer:

                print("Primal infeasibility.\n")
            elif mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")

            c = len(contps) * 2
            n = len(elems) * 3
            if n+c!=len(xx):
                print("Error in the length of xx")
                return None
            contact_forces = xx[-c:]  #
            #print("Duzina c je:", len(contact_forces))

            r = xx[0:n]
            #print("Duzina r je:", len(r))
            result['r'] = r
            result['contact forces'] = contact_forces
            #print(xx)
            #print(result['contact forces'])
            dis=y
            y = y[0:n]
            #print("Displacements are", y)
            #print("Duzina y je", len(dis))
            result["xc"] = xc
            # normalize the displacement
            sum = 0
            element_index = 0
            for k, value in elems.items():
                sum += value.ll[0] * y[element_index * 3] + value.ll[1] * \
                       y[element_index * 3 + 1] + value.ll[2] * y[element_index * 3 + 2]
                element_index += 1
                print(f"element {value.id} has live load {value.ll}")
            print("Sum of displacement for normalization", sum)
            if sum == 0:
                result["displacements"] = y[0:len(elems) * 3]
            else:
                result["displacements"] = (
                        np.array(y[0:len(elems) * 3]) / sum).tolist()
            print("Displacement for update element model", result["displacements"])
            # max_disp = 0
            # for i in range(0, len(elems)):
            #     max_disp = max(max_disp, abs(y[i * 3]), abs(y[i * 3 + 1]))
            #
            #
            # if max_disp == 0:
            #     result["displacements"] = np.zeros(len(elems) * 3).tolist()
            # else:
            #     # the rotation shouldn't be normalized
            #     # y_scaled = np.array(y[0:len(elems)*3])/max_disp*np.sign(sum)
            #     # y_scaled[2::3] = y_scaled[2::3]*max_disp/np.sign(sum)
            #     # result["displacements"] = y_scaled.tolist()
            #     # sum(liveload * displacement) = 1
            #     # sum = sum*1e-3
            #     y_scaled = np.array(y[0:len(elems) * 3]) / sum
            #     print("Normalize displacement by", sum)
            #     # y_scaled[2::3] = y_scaled[2::3]*sum
            #
            #     result["displacements"] = y_scaled.tolist()

            result['suc'] = suc
            _update_elem_disp_2d(contps, elems, result["displacements"])

            _displace_model(elems, contps)
            _update_elem_disp_2d(contps, elems, result["displacements"])

    return r, contact_forces, solsta, y
    #return r, contact_forces

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



# Define the solver for TH 3D Force-based
##############################################################################
# # Ovo je za OMM*delta_x, a ne samo r
# def solve_rigid_associative_TH_3d_f(elems, contps):
#
#     Aglobal = cal_A_global_3d(elems, contps)
#     An, At = split_A_matrix(Aglobal)
#
#     # result container
#     result = dict()
#     inf = 0.0
#     # g = cal_gap_3d(contps)
#     r = len(elems) * 6  # del_x is number of unknowns = n of elem*6DOF
#     c = len(contps)  # c is number of contact points
#     cont_num = 3 * len(contps)  # total number of unknown contact forces
#     n_num = c
#     t_num = 2 * c
#     mu = 0.7  # TODO: For now it is universal, but make it possible to read it for each contps from .csv file
#
#     # Create mass matrices:
#     # Overline mass matrix: #TODO: Napravi za vise elemenata automatski
#     OMM_ground = []
#     OMM_el1 = []
#     mass_matrix_index = 0
#     for key, value in elems.items():
#         if value.type == "ground":
#
#             OMM_ground.extend([[value.omm[0][0], 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, value.omm[1][1], 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, value.omm[2][2], 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, value.omm[3][3], 0.0, 0.0, ],
#                                [0.0, 0.0, 0.0, 0.0, value.omm[4][4], 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, value.omm[5][5]]])
#         else:
#             OMM_el1.extend([[value.omm[0][0], 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, value.omm[1][1], 0.0, 0.0, 0.0, 0.0],
#                             [0.0, 0.0, value.omm[2][2], 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, value.omm[3][3], 0.0, 0.0, ],
#                             [0.0, 0.0, 0.0, 0.0, value.omm[4][4], 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, value.omm[5][5]]])
#
#     OMM_ground = np.array(OMM_ground)
#     # OMM_ground = OMM_ground.T @ OMM_ground
#     OMM_el1 = np.array(OMM_el1)
#     # OMM_el1 = OMM_el1.T @ OMM_el1
#     Z = np.zeros((6, 6), dtype=int)  # Create off-diagonal zeros array
#     OMM_global = np.asarray(np.bmat([[OMM_ground, Z], [Z, OMM_el1]]))
#
#     OMM_global_tr = OMM_global.T @ OMM_global
#
#     # Call gaps:
#     g = []
#     g_n = []
#     g_t = []
#     for key, value in contps.items():
#         #for i in range(2):  # 2variables(t,n)*1nodes*2contact faces
#         g.extend([value.gap[0], value.gap[1], value.gap[2]])
#
#     g_n = g[2::3]
#     g_t = []
#     for i in range(0, len(g), 3):
#         g_t.extend(g[i:i + 2])
#
#     g_n_arr = np.array(g_n)
#     g_t_arr = np.array(g_t)
#     # Call ext load vector:
#     fo = []
#     for element in elems.values():
#         fo.extend([element.totload[0], element.totload[1], element.totload[2], element.totload[3], element.totload[4],
#                    element.totload[5]])
#
#     fo = np.array(fo)
#     # Define the decision variables for del_x and c
#     x = cp.Variable(r)
#     n = cp.Variable(n_num)
#     t = cp.Variable(t_num)
#     # Define the objective function
#
#     obj = cp.Maximize((-1 / 2) * cp.quad_form(x, OMM_global_tr) - g_n @ n - g_t @ t)
#
#
#     # soc_lin_con = [0 <= mu * y[i][0] for i in range(c)] # ako obrisem [0], kako ce znati za koji element vektora c? n?
#     # soc_lin_con = [0 <= mu * y[i] for i in range(c)]
#     #const = 1 / mu
#     # soc_constraints = [cp.SOC(mu * n[i], t[i:i+2]) for i in range(n_num)] # For each contact point
#     soc_constraints = [cp.SOC(mu * n[i], t[i * 2:i * 2 + 2]) for i in range(n_num)]
#     #lin_constraint1 = [mu * n[i] >= 0 for i in range(n_num)]
#     lin_constraint2 = [OMM_global @ x + An @ n + At @ t == fo]
#
#     # Create the problem instance
#     # problem = cp.Problem(obj, soc_constraints + [del_un == At.T @ x, del_ut == An.T @ x])
#     # An_t = An.T
#     # At_t = At.T
#     problem = cp.Problem(obj, soc_constraints + lin_constraint2 )
#
#     # Solve the problem
#     #problem.solve()
#     problem.solve(solver=cp.MOSEK, verbose=True)
#     # Print the solution
#     print("Solution status: ", problem.status)
#     print("Objective value: ", problem.value)
#     print("Solution vector for del_x: ", x.value)
#     print("Solution vector for c: ", n.value)
#     print(t.value)
#     _update_elem_disp_3d(contps, elems, x.value)
#     _displace_model_3d(elems, contps)
#     _update_elem_disp_3d(contps, elems, x.value)
#
#     return x.value, n.value, t.value
#
#
#
#
#    # This is SCS solver. I moved to CVXPY solver that is not commented here
#     #       """Solve time history in 3D using associative flow rule considering infinite compressive strength
#     #
#     # :param elems: Dictionary of elements
#     # :type elems: dict
#     # :param contps: Dictionary of contact points
#     # :type contps: dict
#     # :return: Solution
#     # :rtype: dict
#     # """
#     # Aglobal = cal_A_global_3d(elems, contps)
#     # # result container
#     # result = dict()
#     # inf = 0.0
#     # # g = cal_gap_3d(contps)
#     #
#     # r = len(elems)*6    # r is number of unknowns = n of elem*6DOF
#     # normal = len(contps) # Normal contact force = n of contps
#     # t1 = len(contps)  #  Tangential contact force 1 = n of contps
#     # t2 = len(contps)  # Tangential contact force 2 =  = n of contps
#     # cont_num = 3 * len(contps) #total number of unknown contact forces
#     # mu = 0.7 #TODO: For now it is universal, but make it possible to read it for each contps from .csv file
#     #
#     # # # Mass matrix:
#     # # MM_ground = []
#     # # MM_el1 = []
#     # # mass_matrix_index = 0
#     # # for key, value in elems.items():
#     # #     if value.type == "ground":
#     # #
#     # #         MM_ground.extend(
#     # #             [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     # #              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
#     # #     else:
#     # #
#     # #         MM_el1.extend([[-value.mm[0][0], 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -value.mm[1][1], 0.0, 0.0, 0.0, 0.0],
#     # #                         [0.0, 0.0, -value.mm[2][2], 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -value.mm[3][3], 0.0, 0.0, ],
#     # #                         [0.0, 0.0, 0.0, 0.0, -value.mm[4][4], 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -value.mm[5][5]]])
#     # #
#     # # Z = np.zeros((6, 6), dtype=int)  # Create off-diagonal zeros array
#     # # MM_global = np.asarray(np.bmat([[MM_ground, Z], [Z, MM_el1]]))
#     #
#     #
#     #
#     # # Overline mass matrix: #TODO: Napravi za vise elemenata automatski
#     # OMM_ground =[]
#     # OMM_el1 = []
#     # mass_matrix_index = 0
#     # for key, value in elems.items():
#     #     if value.type == "ground":
#     #
#     #         OMM_ground.extend([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
#     #     else:
#     #         #OMM.extend([3 + mass_matrix_index, 3 + mass_matrix_index + 1, 3 + mass_matrix_index + 2])
#     #         #OMM.extend([3 + mass_matrix_index, 3 + mass_matrix_index + 1, 3 + mass_matrix_index + 2])
#     #         OMM_el1.extend([[value.omm[0][0], 0.0, 0.0, 0.0, 0.0, 0.0], [0.0,value.omm[1][1], 0.0, 0.0, 0.0, 0.0],
#     #                      [0.0, 0.0,value.omm[2][2], 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, value.omm[3][3], 0.0, 0.0,],
#     #                      [0.0, 0.0, 0.0, 0.0,value.omm[4][4], 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, value.omm[5][5]]])
#     #
#     #
#     #
#     # Z = np.zeros((6, 6), dtype=int)  # Create off-diagonal zeros array
#     # OMM_global = np.asarray(np.bmat([[OMM_ground, Z], [Z, OMM_el1]]))
#     # #OMM_np = np.array(OMM)
#     # OMM_global_tr= OMM_global.T
#     #
#     # #Add necessary zeroes to make full matrices
#     # Z1 = np.zeros((cont_num, r), dtype=float)
#     # Z2 = np.zeros((cont_num, cont_num), dtype=float)
#     # Z3 = np.zeros((r, cont_num), dtype=float)
#     # # this merged_OMM_0 is merged matrix consisted of global OMM matrix and zero elements
#     # merged_OMM_0 = np.asarray(np.bmat([[OMM_global_tr, Z3], [Z1, Z2]]))
#     #
#     # # Input for solver - next to the quadratic part:
#     # P = sparse.csc_matrix(merged_OMM_0)
#     #
#     # # c vector is the one that goes with linear part of obj f - go.tr - with contacts TODO: DA LI IDE S MINUSIMA?
#     # #TODO: Proveri gaps OVDE JE PROBLEM
#     #
#     # #cal_gap_3d(contps)
#     # gaps = []
#     # for key, value in contps.items():
#     #     # for i in range(2):  # 2variables(t,n)*1nodes*2contact faces
#     #     gaps.extend([value.gap[0], value.gap[1], value.gap[2]])
#     #
#     #
#     # c = np.hstack([np.zeros(r), gaps])  #
#     #
#     # # A matrix
#     # # zero cone:
#     # merged_1_Agl = np.concatenate((OMM_global, Aglobal), axis=1)
#     # # SOC: #Mislim da ovako treba da napises Aexp da bi sacuvali mesta, indices
#     # A_r = np.zeros((cont_num, r), dtype=float)
#     # A_normals = np.zeros((cont_num, normal), dtype=float)
#     # A_t1_t2_16 = np.eye((t1 + t2), dtype=float)
#     # np.fill_diagonal(A_t1_t2_16, -1)
#     # A_t1_t2_8 = np.zeros((t1, t1 + t2), dtype=float)
#     # A_t1_t2 = np.concatenate((A_t1_t2_16, A_t1_t2_8))
#     # A_SOC = np.concatenate((A_r, A_t1_t2, A_normals), axis=1)
#     #
#     # #b vector -- what is on the other side of equations of cones
#     # #b zero cone
#     # b_zero = []
#     # for element in elems.values():
#     #     b_zero.extend([element.totload[0],element.totload[1],element.totload[2],element.totload[3],element.totload[4], element.totload[5]])
#     #
#     # #b SOC:
#     # b_SOC = np.zeros(cont_num)
#     # for i in range(normal):
#     #     b_SOC[i * 3 + 2] = 1 * mu
#     #
#     # b = np.hstack([b_zero, b_SOC])
#     #
#     # # Assemble A matrix - A_zero + A_SOC
#     # A = sparse.vstack(
#     #     [
#     #         # zero cone
#     #         sparse.hstack([sparse.csc_matrix(merged_1_Agl)]), #da li ovde treba minus
#     #         # positive cone - NEMAS
#     #         # exponential cones
#     #         sparse.hstack([sparse.csc_matrix(A_SOC)]),
#     #     ],
#     #     format="csc",
#     # )
#     # # Populate dicts with data to pass into SCS
#     # data = dict(P=P, A=A, b=b, c=c)
#     # cone = dict(z=r, l=0, q=cont_num, qsize=1)
#     #
#     # # Initialize solver
#     # solver = scs.SCS(
#     #     data,
#     #     cone,
#     #
#     #     eps_abs=1e-7,
#     #     eps_rel=1e-7,
#     # )
#     # # Solve!
#     # sol = solver.solve()
#     #
#     # print(f"SCS took {sol['info']['iter']} iters")
#     # print("Optimal solution vector x*:")
#     # print(sol["x"])
#     # n = len(elems) * 6
#     # displ_x = sol["x"][0:n]
#     # result['displacements'] = displ_x
#     #
#     # print("Optimal dual vector y*:")
#     # print(sol["y"])
#     #
#     # print("Optimal dual vector s*:")
#     # print(sol["s"])
#     #
#     # # Update 3D element - with displ
#     # # normalize the displacement
#     #
#     #
#     #
#     # _update_elem_disp_3d(contps, elems, result["displacements"])
#     # _displace_model_3d(elems, contps)
#     # _update_elem_disp_3d(contps, elems, result["displacements"])
#     #
#     # with open('coeff_optimization.txt', 'w') as f:
#     #     f.write('Aglobal' + '\n' + str(Aglobal) + '\n' )
#     #     f.write('OMM_global (overline mass matrices of elems)'+ '\n' + str(OMM_global) + '\n')
#     #     f.write('OMM_global merged with zeroes' + '\n' + str(merged_OMM_0) + '\n')
#     #     f.write('P-Sparse coeff. with quadratic part of obj.f.' + '\n' + str(P) + '\n')
#     #     f.write('c-coeff. with lin. part of obj.f.(zeroes+gaps)' + '\n' + str(c) + '\n')
#     #     f.write('b-cones equal to..(fo+SOC)' + '\n' + str(b) + '\n')
#     #     f.write('Zero cone coeff.(merged_1_Agl)' + '\n' + str(merged_1_Agl) + '\n')
#     #     f.write('SOC coeff.(A_SOC)' + '\n' + str(A_SOC) + '\n')
#     #     f.write('A-Sparse coeff. with coeff. of cones(constraints)' + '\n' + str(A) + '\n')
#     #     f.write('Displ. at center' + '\n' + str(displ_x) + '\n')
#     # return sol["x"]
#
#
# def _displace_model_3d(elems, contps):
#     # update vertices information because the next step could fail
#     # element vertices
#     for key, value in elems.items():
#         vertices = np.array(value.vertices)
#         center = np.asarray(value.center)
#         vertices_res_center = vertices-center
#         rot_angles = np.asarray(value.displacement[3:])
#         rotated_vertices_res_center = rotate_3d(
#             vertices_res_center, rot_angles, order='xyz')
#         disp_center = np.asarray(value.displacement[:3])
#         new_vertices = rotated_vertices_res_center+disp_center+center
#         value.vertices = new_vertices.tolist()
#
#     for k, value in contps.items():
#         disp_center = np.asarray(elems[value.cand].displacement[:3])
#         center = np.asarray(elems[value.cand].center)
#         point_coord = np.asarray(value.coor)
#         point_coord_res_center = point_coord-center
#         rot_angles = np.asarray(elems[value.cand].displacement[3:])
#         rotated_point_coord_res_center = rotate_3d(
#             np.expand_dims(point_coord_res_center, axis=0), rot_angles)[0]
#         new_point_coord = rotated_point_coord_res_center+disp_center+center
#         value.coor = new_point_coord.tolist()
#         value.displacement = [0, 0, 0]
#
#     for key, value in elems.items():
#         value.center[0] = value.center[0]+value.displacement[0]
#         value.center[1] = value.center[1]+value.displacement[1]
#         value.center[2] = value.center[2]+value.displacement[2]
#         value.displacement = [0, 0, 0, 0, 0, 0]
#
#
#
# ########################################################################
# # # Ovo je sa r
# # def solve_rigid_associative_TH_3d_f(elems, contps):
# #     """Solve time history in 3D using associative flow rule considering infinite compressive strength
# #
# #     :param elems: Dictionary of elements
# #     :type elems: dict
# #     :param contps: Dictionary of contact points
# #     :type contps: dict
# #     :return: Solution
# #     :rtype: dict
# #     """
# #
# #
# #     getcontext().prec = 28
# #     Aglobal = cal_A_global_3d(elems, contps)
# #     # result container
# #     result = dict()
# #     inf = 0.0
# #     #g = cal_gap_3d(contps)
# #
# #     r = len(elems) * 6  # r is number of unknowns = n of elem*6DOF
# #     normal = len(contps)  # Normal contact force = n of contps
# #     t1 = len(contps)  # Tangential contact force 1 = n of contps
# #     t2 = len(contps)  # Tangential contact force 2 =  = n of contps
# #     cont_num = 3 * len(contps)  # total number of unknown contact forces
# #     mu = 0.65  # TODO: For now it is universal, but make it possible to read it for each contps from .csv file
# #
# #     # Mass matrix:
# #     MM_ground = []
# #     MM_el1 = []
# #     mass_matrix_index = 0
# #     for key, value in elems.items():
# #         if value.type == "ground":
# #
# #             MM_ground.extend(
# #                 [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# #                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
# #         else:
# #
# #             MM_el1.extend([[-value.mm[0][0], 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -value.mm[1][1], 0.0, 0.0, 0.0, 0.0],
# #                            [0.0, 0.0, -value.mm[2][2], 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -value.mm[3][3], 0.0, 0.0, ],
# #                            [0.0, 0.0, 0.0, 0.0, -value.mm[4][4], 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -value.mm[5][5]]])
# #
# #     Z = np.zeros((6, 6), dtype=int)  # Create off-diagonal zeros array
# #     MM_global = np.asarray(np.bmat([[MM_ground, Z], [Z, MM_el1]]))
# #
# #     # Overline mass matrix: #TODO: Napravi za vise elemenata automatski
# #     OMM_ground = []
# #     OMM_el1 = []
# #     mass_matrix_index = 0
# #     for key, value in elems.items():
# #         if value.type == "ground":
# #             # OMM.extend([3 + mass_matrix_index, 3 + mass_matrix_index + 1, 3 + mass_matrix_index + 2])
# #             # OMM.extend([3 + mass_matrix_index, 3 + mass_matrix_index + 1, 3 + mass_matrix_index + 2])
# #             OMM_ground.extend(
# #                 [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# #                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
# #         else:
# #             # OMM.extend([3 + mass_matrix_index, 3 + mass_matrix_index + 1, 3 + mass_matrix_index + 2])
# #             # OMM.extend([3 + mass_matrix_index, 3 + mass_matrix_index + 1, 3 + mass_matrix_index + 2])
# #             OMM_el1.extend([[value.omm[0][0], 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, value.omm[1][1], 0.0, 0.0, 0.0, 0.0],
# #                             [0.0, 0.0, value.omm[2][2], 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, value.omm[3][3], 0.0, 0.0, ],
# #                             [0.0, 0.0, 0.0, 0.0, value.omm[4][4], 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, value.omm[5][5]]])
# #
# #     Z = np.zeros((6, 6), dtype=int)  # Create off-diagonal zeros array
# #     OMM_el1_inv = np.linalg.inv(OMM_el1)
# #     OMM_global = np.asarray(np.bmat([[OMM_ground, Z], [Z, OMM_el1_inv]]))
# #     # OMM_np = np.array(OMM)
# #     #OMM_global_inv = np.linalg.inv(OMM_global)
# #
# #     # Add necessary zeroes to make full matrices
# #     Z1 = np.zeros((cont_num, r), dtype=float)
# #     Z2 = np.zeros((cont_num, cont_num), dtype=float)
# #     Z3 = np.zeros((r, cont_num), dtype=float)
# #     # this merged_OMM_0 is merged matrix consisted of global OMM matrix and zero elements
# #     merged_OMM_0 = np.asarray(np.bmat([[OMM_global, Z3], [Z1, Z2]]))
# #
# #     # Input for solver - next to the quadratic part:
# #     P = sparse.csc_matrix(merged_OMM_0)
# #
# #     # c vector is the one that goes with linear part of obj f - go.tr - with contacts TODO: DA LI IDE S MINUSIMA?
# #     # TODO: Proveri gaps OVDE JE PROBLEM
# #
# #     cal_gap_3d(contps)
# #     gaps = []
# #     for key, value in contps.items():
# #         # for i in range(2):  # 2variables(t,n)*1nodes*2contact faces
# #         gaps.extend([value.gap[0], value.gap[1], value.gap[2]])
# #
# #     c = np.hstack([np.zeros(r), gaps])  #
# #
# #     # A matrix
# #     # zero cone:
# #     Ones_mat = np.eye(r, dtype=float)
# #     merged_1_Agl = np.concatenate((Ones_mat, Aglobal), axis=1)
# #     # SOC: #Mislim da ovako treba da napises Aexp da bi sacuvali mesta,indices
# #     A_r = np.zeros((cont_num, r), dtype=float)
# #     A_normals = np.zeros((cont_num, normal), dtype=float)
# #     A_t1_t2_16 = np.eye((t1 + t2), dtype=float)
# #     np.fill_diagonal(A_t1_t2_16, -1)
# #     A_t1_t2_8 = np.zeros((t1, t1 + t2), dtype=float)
# #     A_t1_t2 = np.concatenate((A_t1_t2_16, A_t1_t2_8))
# #     A_SOC = np.concatenate((A_r, A_t1_t2, A_normals), axis=1)
# #
# #     # b vector -- what is on the other side of equations of cones
# #     # b zero cone
# #     b_zero = []
# #     for element in elems.values():
# #         b_zero.extend(
# #             [element.totload[0], element.totload[1], element.totload[2], element.totload[3], element.totload[4],
# #              element.totload[5]])
# #
# #     # b SOC:
# #     b_SOC = np.zeros(cont_num)
# #     for i in range(normal):
# #         b_SOC[i * 3 + 2] = 1 * mu
# #
# #     b = np.hstack([b_zero, b_SOC])
# #
# #     # Assemble A matrix - A_zero + A_SOC
# #     A = sparse.vstack(
# #         [
# #             # zero cone
# #             sparse.hstack([sparse.csc_matrix(merged_1_Agl)]),
# #             # positive cone - NEMAS
# #             # exponential cones
# #             sparse.hstack([sparse.csc_matrix(A_SOC)]),
# #         ],
# #         format="csc",
# #     )
# #     # Populate dicts with data to pass into SCS
# #     data = dict(P=P, A=A, b=b, c=c)
# #     cone = dict(z=r, l=0, q=cont_num, qsize=1)
# #
# #     # Initialize solver
# #     solver = scs.SCS(
# #         data,
# #         cone,
# #         eps_abs=1e-7,
# #         eps_rel=1e-7,
# #     )
# #     # Solve!
# #     sol = solver.solve()
# #
# #     print(f"SCS took {sol['info']['iter']} iters")
# #     print("Optimal solution vector x*:")
# #     print(sol["x"])
# #
# #     print("Optimal dual vector y*:")
# #     print(sol["y"])
# #
# #     print("Optimal dual vector s*:")
# #     print(sol["s"])
# #
# #     return sol["x"]