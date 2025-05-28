import math
from ..calc.a import cal_A_global_3d, cal_A_global_2d
import sys
import mosek
import numpy as np
from ..utils.parameter import get_dimension
from .util import _update_elem_disp_2d, _update_elem_disp_3d

print_detail = False


def solve_infinitefc_associative(elems, contps):
    """Solve limit force using associative flow rule considering infinite compressive strength

    :param elems: Dictionary of elements
    :type elems: dict
    :param contps: Dictionary of contact points
    :type contps: dict
    :return: Solution
    :rtype: dict
    """
    if get_dimension() == 2:
        return solve_infinitefc_associative_2d(elems, contps)
    elif get_dimension() == 3:
        return solve_infinitefc_associative_3d(elems, contps)


def solve_infinitefc_associative_3d(elems, contps):
    """Solve the model with inifite fc, frictional contact and associative flow rule

    :param elems: Dictionary of elements
    :type elems: dict
    :param contps: Dictionary of contact points
    :type contps: dict
    :return: Solution. Available keys are 'displacements', 'limit_force', and 'contact_forces'
    :rtype: dict
    """
    result = dict()
    Aglobal = cal_A_global_3d(elems, contps)

    inf = 0.0

    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    with mosek.Env() as env:
        # Create a task object
        with env.Task(0, 0) as task:
            # Attach a log stream printer to the task
            if print_detail:
                task.set_Stream(mosek.streamtype.log, streamprinter)

            # Bound keys and values for constraints
            bkc = []
            blc = []
            buc = []
            for element in elems.values():
                if element.type == "ground":
                    bkc.extend([mosek.boundkey.fr,
                                mosek.boundkey.fr,
                                mosek.boundkey.fr,
                                mosek.boundkey.fr,
                                mosek.boundkey.fr,
                                mosek.boundkey.fr])
                    blc.extend([-inf, -inf, -inf, -inf, -inf, -inf])
                    buc.extend([inf, inf, inf, inf, inf, inf])
                else:
                    bkc.extend([mosek.boundkey.fx,
                                mosek.boundkey.fx,
                                mosek.boundkey.fx,
                                mosek.boundkey.fx,
                                mosek.boundkey.fx,
                                mosek.boundkey.fx])
                    blc.extend([element.dl[0],
                                element.dl[1], element.dl[2], element.dl[3],
                                element.dl[4], element.dl[5]])
                    buc.extend([element.dl[0],
                                element.dl[1], element.dl[2], element.dl[3],
                                element.dl[4], element.dl[5]])
            for point in contps.values():  # 4th variable
                bkc.extend([mosek.boundkey.fx])
                blc.extend([0])
                buc.extend([0])

            # Bound keys for variables
            bkx = []
            blx = []
            bux = []
            for key, value in contps.items():
                bkx.extend([mosek.boundkey.fr, mosek.boundkey.fr,
                           mosek.boundkey.lo, mosek.boundkey.lo])
                blx.extend([-inf, -inf, 0, 0])
                bux.extend([+inf, +inf, +inf, +inf])
                # for i in range(4):
                #     bkx.append(mosek.boundkey.fr)
                #     blx.append(-inf)
                #     bux.append(+inf)

            bkx.append(mosek.boundkey.lo)
            blx.append(0)
            bux.append(+inf)

            # Objective coefficients
            c = []
            for key, value in contps.items():
                # for i in range(2):  # 2variables(t,n)*1nodes*2contact faces
                c.extend([-value.gap[0], -value.gap[1], -value.gap[2], 0])
                # print(-g[g_index])
            c.append(1.0)

            # Below is the sparse representation of the A
            # matrix stored by column.
            asub = []
            aval = []
            for i, point in enumerate(contps.values()):
                for j in range(3):
                    col = i*4+j
                    col_A = i*3+j
                    col_index = []
                    col_value = []
                    for element_id in range(len(elems)):
                        for equ in range(6):
                            row_A = element_id*6+equ
                            row = element_id*6+equ
                            if Aglobal[row_A][col_A] != 0:
                                col_index.append(row)
                                col_value.append(Aglobal[row_A][col_A])
                    if j == 2:  # add extra 4th variable for each contact point
                        # if point.cont_type.mu > 0:
                        # col_index.append(len(elems)*6+i)
                        # col_value.append(point.cont_type.mu)
                        # asub.extend([col_index, [len(elems)*6+i]])
                        # aval.extend([col_value, [-1]])

                        col_index.append(len(elems)*6+i)
                        col_value.append(point.cont_type.mu)
                        asub.extend([col_index, [len(elems)*6+i]])
                        aval.extend([col_value, [-1]])
                    else:
                        asub.append(col_index)
                        aval.append(col_value)

            col_index = []
            col_value = []
            i = 0
            for element in elems.values():
                col_index.extend([6*i, 6*i+1, 6*i+2, 6*i+3, 6*i+4, 6*i+5])
                col_value.extend(
                    [-element.ll[0], -element.ll[1], -element.ll[2], -element.ll[3], -element.ll[4], -element.ll[5]])
                i += 1
            asub.append(col_index)
            aval.append(col_value)

            # define the optimization task
            numvar = len(bkx)
            numcon = len(bkc)
            task.appendcons(numcon)
            task.appendvars(numvar)
            for j in range(numvar):
                task.putcj(j, c[j])
                task.putvarbound(j, bkx[j], blx[j], bux[j])
                task.putacol(j, asub[j], aval[j])
            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])
            for i in range(len(contps)):
                task.appendcone(mosek.conetype.quad,
                                0.0,
                                [4*i+3, 4*i+0, 4*i+1])
            task.putobjsense(mosek.objsense.maximize)

            # Solve the problem
            task.writedata("data.opf")
            #task.putdouparam(dparam.intpnt_co_tol_dfeas, 1.0e-8)

            #from mosek import dparam
            #task.putdouparam(dparam.intpnt_co_tol_mu_red, 0)
            task.optimize()

            if print_detail:
                task.solutionsummary(mosek.streamtype.msg)

            # Get status information about the solution
            #prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)

            xx = [0.] * numvar
            xc = [0.]*numcon
            y = [0.]*numcon
            suc = [0.]*numcon

            # !!!!!!!!!!!!!!!!unknow also yield to correct result -> from experience
            # if (solsta == mosek.solsta.optimal or solsta == mosek.solsta.unknown):
            if (solsta == mosek.solsta.optimal):
                task.getxx(mosek.soltype.itr, xx)
                task.getxc(mosek.soltype.itr, xc)
                task.gety(mosek.soltype.itr, y)
                task.getsuc(mosek.soltype.itr, suc)
                # if print_detail:
                #     print("Optimal solution: ")
                #     for i in range(numvar):
                #         print("x[" + str(i) + "]=" + str(xx[i]))
                print(f'limit force is {xx[-1]}')

            else:
                print("Other solution status")

            result["limit_force"] = xx[-1]
            result["contact_forces"] = xx[0:numvar-1]
            result["xc"] = xc
            result['suc'] = suc
            # normalize the displacement
            sum = 0
            element_index = 0
            for k, value in elems.items():
                sum += value.ll[0]*y[element_index*6]+value.ll[1] * \
                    y[element_index*6+1]+value.ll[2]*y[element_index*6+2]\
                    + value.ll[3]*y[element_index*6+3]+value.ll[4]*y[element_index*6+4]\
                    + value.ll[5]*y[element_index*6+5]
                element_index += 1
            if sum == 0:
                result["displacements"] = y[0:len(elems)*6]
            else:
                result["displacements"] = (
                    np.array(y[0:len(elems)*6])/sum).tolist()
            if result['limit_force'] > 0:
                _update_elem_disp_3d(contps, elems, result["displacements"])
    return result


def solve_infinitefc_associative_2d(elems, contps):
    """Solve limit force in 2D using associative flow rule considering infinite compressive strength

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

    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    limit_force = 0
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
                    blc.extend([value.dl[0], value.dl[1], value.dl[2]])
                    buc.extend([value.dl[0], value.dl[1], value.dl[2]])

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
            for key, value in contps.items():
                for i in range(2):
                    bkx.append(mosek.boundkey.fr)
                    blx.append(-inf)
                    bux.append(+inf)

                    g_index += 1
            bkx.append(mosek.boundkey.lo)
            blx.append(0)
            bux.append(+inf)

            # Objective coefficients
            c = []
            for key, value in contps.items():
                # for i in range(2):  # 2variables(t,n)*1nodes*2contact faces
                c.extend([-value.gap[0], -value.gap[1]])
                # print(-g[g_index])
            c.append(1.0)

            # Below is the sparse representation of the A
            # matrix stored by column.
            asub = []
            aval = []

            for i, value in enumerate(contps.values()):
                for j in range(2):  # 2variables(t,n)*1nodes*
                    col = i*2+j
                    col_index = []
                    col_value = []
                    for row in range(len(elems)*3):
                        if Aglobal[row][col] != 0:
                            col_index.append(row)
                            col_value.append(Aglobal[row][col])
                    _start_row = len(elems)*3 + math.floor(col/2)*3
                    col_index.extend(
                        list(range(_start_row, _start_row+3)))
                    if col % 2 == 0:
                        col_value.extend([1, -1, 0])
                    else:
                        col_value.extend(
                            [-value.cont_type.mu, -value.cont_type.mu, -1])

                    asub.append(col_index)
                    aval.append(col_value)

            col_index = []
            col_value = []
            i = 0
            for key, value in elems.items():
                col_index.extend([3*i, 3*i+1, 3*i+2])
                col_value.extend(
                    [-value.ll[0], -value.ll[1], -value.ll[2]])
                i += 1
            asub.append(col_index)
            aval.append(col_value)

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
                task.putacol(j,                  # Variable (column) index.
                             # Row index of non-zeros in column j.
                             asub[j],
                             aval[j])            # Non-zero Values of column j.

            # Set the bounds on constraints.
             # blc[i] <= constraint_i <= buc[i]

            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)

            # Solve the problem
            # task.putintparam(mosek.iparam.optimizer,
            #                  mosek.optimizertype.intpnt)
            task.putintparam(mosek.iparam.optimizer,
                             mosek.optimizertype.dual_simplex)
            # task.putintparam(mosek.iparam.intpnt_max_iterations, 1000)
            # task.putintparam(mosek.iparam.intpnt_starting_point,
            #                  mosek.startpointtype.constant)
            # task.putintparam(mosek.iparam.intpnt_solve_form,
            #                  mosek.solveform.dual)
            # task.putintparam(mosek.iparam.bi_clean_optimizer,
            #                  mosek.optimizertype.dual_simplex)
            #task.putintparam(mosek.iparam.num_threads, 4)
            # task.putintparam(mosek.iparam.presolve_eliminator_max_num_tries,
            #                  0)
            task.putintparam(mosek.iparam.presolve_use,
                             mosek.presolvemode.off)
            task.writedata("data.opf")
            task.optimize()
            if print_detail:
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.log)

            # Get status information about the solution
            #solsta = task.getsolsta(mosek.soltype.bas)
            solsta = task.getsolsta(mosek.soltype.bas)
            #solsta_itr = task.getsolsta(mosek.soltype.itr)

            xx = [0.] * numvar
            xc = [0.]*numcon

            y = [0.]*numcon
            suc = [0.]*numcon
            if (solsta == mosek.solsta.optimal):

                task.getxx(mosek.soltype.bas,  # Request the basic solution.
                           xx)
                task.getxc(mosek.soltype.bas, xc)
                task.gety(mosek.soltype.bas, y)
                task.getsuc(mosek.soltype.bas, suc)
                # if print_detail:
                #     print("Optimal solution: ")
                #     for i in range(numvar):
                #         print("x[" + str(i) + "]=" + str(xx[i]))
                #     print("y")
                #     for i in range(numcon):
                #         print("y[" + str(i) + "]=" + str(y[i]))
                limit_force = xx[-1]
            # elif (solsta_itr == mosek.solsta.optimal):

            #     task.getxx(mosek.soltype.itr,  # Request the basic solution.
            #                xx)
            #     task.getxc(mosek.soltype.itr, xc)
            #     task.gety(mosek.soltype.itr, y)
            #     task.getsuc(mosek.soltype.itr, suc)
            #     # if print_detail:
            #     #     print("Optimal solution: ")
            #     #     for i in range(numvar):
            #     #         print("x[" + str(i) + "]=" + str(xx[i]))
            #     #     print("y")
            #     #     for i in range(numcon):
            #     #         print("y[" + str(i) + "]=" + str(y[i]))
            #     limit_force = xx[-1]
            else:
                if print_detail:
                    print("Other solution status")
                # return 0,[0.] * numvar
                limit_force = 0
        result["limit_force"] = limit_force
        result["contact_forces"] = xx[0:numvar-1]
        # dual optimization solutions

        # task.getsolutionslice(
        #     mosek.soltype.bas, mosek.solbasm.y, 0, len(elems)*3, y)

        result["xc"] = xc
        # normalize the displacement
        sum = 0
        element_index = 0
        for k, value in elems.items():
            sum += value.ll[0]*y[element_index*3]+value.ll[1] * \
                y[element_index*3+1]+value.ll[2]*y[element_index*3+2]
            element_index += 1
        if sum == 0:
            result["displacements"] = y[0:len(elems)*3]
        else:
            result["displacements"] = (
                np.array(y[0:len(elems)*3])/sum).tolist()

        result['suc'] = suc
        if result['limit_force'] > 0:
            _update_elem_disp_2d(contps, elems, result["displacements"])
    return result
