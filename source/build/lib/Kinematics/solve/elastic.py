import tqdm
import math
import sys
import mosek
import numpy as np


print_detail = False



class ContFace():
    """Contact face class
    """

    def __init__(self, id, height, fc, ft=0):
        """
        :param id: ID of the face
        :type id: int
        :param height: Height of the face
        :type height: float
        :param fc: Compressive strength
        :type fc: float
        """
        self.id = id
        self.contps = []
        self.height = height
        self.fc = fc
        self.ft = ft

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False
def _line1(sigma_c, sigma_t, h_j):
    a = (3*h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (3*h_j ** 2*sigma_t)/8
    return (-2*a, -c)


def _line2(sigma_c, sigma_t, h_j):
    a = (h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (3*h_j ** 2*(sigma_c - sigma_t))/16 - \
        (h_j ** 2*(sigma_c - 3*sigma_t))/8
    return (-2*a, -c)


def _line3(sigma_c, sigma_t, h_j):
    a = -(h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (h_j ** 2*(3*sigma_c - sigma_t))/8 - \
        (3*h_j ** 2*(sigma_c - sigma_t))/16
    return (-2*a, -c)


def _line4(sigma_c, sigma_t, h_j):
    a = -(3*h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (3*h_j ** 2*sigma_c)/8
    return (-2*a, -c)


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

def solve_elastic_finitefc_associative_2d(elems, contps, Aglobal=None,thickness_dict=None,material_dict=None):

    # result container
    result = dict()
    # assemble contact faces
    contfs = dict()
    for p in tqdm.tqdm(contps.values(), desc='assemble contact faces'):
        if p.faceID not in contfs.keys():
            face = ContFace(p.faceID, p.section_h,
                            p.cont_type.fc, p.cont_type.ft)
            contfs[face.id] = face
            contfs[p.faceID].contps.append(p.id)
        else:
            contfs[p.faceID].contps.append(p.id)

    nb_contfs = len(contfs)
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
            for key, value in tqdm.tqdm(elems.items(), desc='Bound keys and values for constraints on equilbrium'):
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
            for key, value in tqdm.tqdm(contps.items(), desc='Bound keys and values for constraints on sliding'):
                if value.cont_type.type == 'friction' or value.cont_type.type == 'friction_fc':
                    for i in range(3):
                        bkc.append(mosek.boundkey.up)
                        blc.append(-inf)
                        buc.append(0.0)
                elif value.cont_type.type == 'friction_fc_cohesion':
                    Ft = value.cont_type.ft*value.section_h/4
                    Fc = value.cont_type.fc*value.section_h/4
                    bkc.extend(
                        [mosek.boundkey.up, mosek.boundkey.up, mosek.boundkey.up, mosek.boundkey.lo])
                    blc.extend([-inf, -inf, -inf, -Fc])
                    buc.extend(
                        [value.cont_type.cohesion, value.cont_type.cohesion, Ft,inf])
                    # Ft = value.cont_type.ft*value.section_h/4
                    # Fc = value.cont_type.fc*value.section_h*1000
                    # bkc.extend(
                    #     [mosek.boundkey.up, mosek.boundkey.up, mosek.boundkey.up])
                    # blc.extend([-inf, -inf, -inf])
                    # buc.extend(
                    #     [value.cont_type.cohesion, value.cont_type.cohesion, Ft])
                else:
                    raise NameError("unknown contact type!")
            # # Bound keys and values for constraints -- crushing failure condition
            # for key, value in tqdm.tqdm(contfs.items(), desc='Bound keys and values for constraints on fc'):
            #     bkc.extend([mosek.boundkey.up, mosek.boundkey.up,
            #                 mosek.boundkey.up, mosek.boundkey.up,
            #                 mosek.boundkey.up, mosek.boundkey.up,
            #                 mosek.boundkey.up, mosek.boundkey.up])
            #     blc.extend([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf])
            #     # buc.extend([0.0, 0.0, (1/16)*value.fc*(value.height**2), (1/16)*value.fc*(value.height**2),
            #     #             (3/16)*value.fc*(value.height**2), (3/16) *
            #     #             value.fc*(value.height**2),
            #     #             (3/8)*value.fc*(value.height**2), (3/8)*value.fc*(value.height**2)])
            #     buc.extend([-_line1(value.fc, value.ft, value.height)[1], -_line1(value.fc, value.ft, value.height)[1], -_line2(value.fc, value.ft, value.height)[1], -_line2(value.fc, value.ft, value.height)[1],
            #                 -_line3(value.fc, value.ft, value.height)[1], -_line3(
            #                     value.fc, value.ft, value.height)[1],
            #                 -_line4(value.fc, value.ft, value.height)[1], -_line4(value.fc, value.ft, value.height)[1]])

            # Bound keys for variables
            bkx = []
            blx = []
            bux = []
            #g_index = 0
            for key, value in tqdm.tqdm(contps.items(), desc='bound key for variables'):
                for i in range(2):
                    bkx.append(mosek.boundkey.fr)
                    blx.append(-inf)
                    bux.append(+inf)

                    #g_index += 1
            #bkx.append(mosek.boundkey.lo)
            #blx.append(0.0)
            bux.append(+inf)

            # # Objective coefficients
            # c = []
            # g_index = 0
            # for key, value in contps.items():
            #     for i in range(2):  # 2variables(t,n)*1nodes*2contact faces
            #         c.append(-0)
            #         # print(-g[g_index])
            #         g_index += 1
            # c.append(1.0)

            # Objective coefficients
            c = []
            for key, value in tqdm.tqdm(contps.items(), desc='Objective coefficients'):
                # for i in range(2):  # 2variables(t,n)*1nodes*2contact faces
                c.extend([-value.gap[0], -value.gap[1]])
                # print(-g[g_index])
            #c.append(1.0)

            # Below is the sparse representation of the A
            # matrix stored by column.
            asub = []
            aval = []
            faceIDs = list(contfs.keys())
            for i, value in tqdm.tqdm(enumerate(contps.values()), desc='assemble A matrix'):
                for j in range(2):  # 2variables(t,n)*1nodes*
                    col = i*2+j
                    col_index = []
                    col_value = []
                    if type(Aglobal) is tuple:  # sparse matrix
                        col_index.extend(Aglobal[0][col])
                        col_value.extend(Aglobal[1][col])
                    else:
                        for row in range(len(elems)*3):
                            if Aglobal[row][col] != 0:
                                col_index.append(row)
                                col_value.append(Aglobal[row][col])
                    _start_row = len(elems)*3 + math.floor(col/2)*4
                    col_index.extend(
                        list(range(_start_row, _start_row+4)))
                    if col % 2 == 0:
                        col_value.extend([1, -1, 0,0])
                    else:
                        col_value.extend(
                            [-value.cont_type.mu, -value.cont_type.mu, -1,-1])

                    asub.append(col_index)
                    aval.append(col_value)

            # col_index = []
            # col_value = []
            # i = 0
            # for key, value in tqdm.tqdm(elems.items(), desc='apply load'):
            #     col_index.extend([3*i, 3*i+1, 3*i+2])
            #     col_value.extend(
            #         [-value.ll[0], -value.ll[1], -value.ll[2]])
            #     i += 1
            # asub.append(col_index)
            # aval.append(col_value)

            numvar = len(bkx)
            numcon = len(bkc)

            # Append 'numcon' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(numcon)

            # Append 'numvar' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(numvar)

            for j in tqdm.tqdm(range(numvar), desc='input var data to mosek'):
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

            for i in tqdm.tqdm(range(numcon), desc='input con data to mosek'):
                task.putconbound(i, bkc[i], blc[i], buc[i])

            # Set up and input quadratic objective
            qsubi = []
            qsubj = []
            qval = []
            cont_index = 0
            for key, value in contps.items():
                Estone = material_dict['E_stone']
                Emortar = material_dict['E_mortar']
                Poissonstone = material_dict['Poisson_stone']
                Poissonmortar = material_dict['Poisson_mortar']
                thickness = thickness_dict[value.id]
                if elems[value.cand].type.startswith('stone'):
                    E = Estone
                    lamda = Poissonstone
                elif elems[contps[value.counterPoint].cand].type.startswith('stone'):
                    E = Emortar
                    lamda = Poissonmortar
                elif elems[value.cand].type.startswith('mortar') and elems[contps[value.counterPoint].cand].type.startswith('mortar'):
                    E = Emortar
                    lamda = Poissonmortar
                else:
                    E = Estone
                    lamda = Poissonstone
                # E = 310e3
                # lamda = 0.2
                kjn = E/thickness#approximation
                #kjn = 1e-1
                kn = kjn*value.section_h*material_dict['wall_thickness']*material_dict['scale_to_m_x']/4
                #kn = 1e8
                kt = kn/(2*(1+lamda))
                # if value.cont_type.ft <=1e-3:
                #     kt = 1e-5
                #     kn = 1e-5
                

                # if elems[value.cand].type.startswith('stone') or elems[contps[value.counterPoint].cand].type.startswith('stone'):
                #     kn*=50
                #     kt*=50
                #kt = 0.33*kn
                qsubi.extend([cont_index, cont_index+1])
                qsubj.extend([cont_index, cont_index+1])
                qval.extend([-1/kt, -1/kn])
                cont_index += 2
            task.putqobj(qsubi, qsubj, qval)

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.maximize)
            print("Start optimization")
            task.optimize()
            if print_detail:
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.log)

            # Get status information about the solution
            #solsta_bas = task.getsolsta(mosek.soltype.bas)
            solsta_itr = task.getsolsta(mosek.soltype.itr)
            convergence = False

            xx = [0.] * numvar
            y = [0.]*numcon
            
            if(solsta_itr == mosek.solsta.optimal):
                task.getxx(mosek.soltype.itr,  # Request the interior-point solution.
                           xx)
                task.gety(mosek.soltype.itr, y)
                convergence = True
                
            else:
                print("Other solution status")

        #result["limit_force"] = limit_force
        result['convergence'] = convergence
        result["contact_forces"] = xx[0:numvar]
        result["displacements"] = y[0:len(elems)*3]
        result["faceIDs"] = faceIDs
        result["contfs"]  =contfs
        #correc the sign of displacement
        for y_index in range(len(result["displacements"])):
            result["displacements"][y_index] = -result["displacements"][y_index]


        _update_elem_disp_2d(contps, elems, result["displacements"])
    return result