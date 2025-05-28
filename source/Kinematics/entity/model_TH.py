# # TODO: change from_json_3d contact parameters type
import pandas
import csv
import os
import json

import numpy as np
from .conttype import ContType
from ..utils.geometry import normalize
from .contpoint import ContPoint
from .element import Element
from ..utils.parameter import get_dimension, set_dimension
from ..calc.anta_id import cal_anta_id


class Model_TH():
    """A container of the model, including all i/o methods
    """

    def __init__(self, elems=dict(), contps=dict()):
        """Constructor method

        :param elems: Dictionary of elements. Key is the element id, value is Element. Defaults to empty dict()
        :type elems: dictionary, optional
        :param contps: Dictionary of contact points. Key is the point id, value is ContPoint. Defaults to empty dict()
        :type contps: dictionry, optional
        """
        self.elems = elems
        self.contps = contps

    def from_json(self, data_dir):
        """Load elements and contact points from json file based input
        The data directory should be arranged as the following:
        - data_dir
        -- element0
        --- geometry.txt
        --- property.txt
        -- element1
        --- geometry.txt
        --- property.txt
        ...
        The geometry file includes the coordinate, normal, tangent of the contact points. The property file includes the material of the element


        :param data_dir: Path to data directory
        :type data_dir: str
        """
        if get_dimension() == 2:
            self.elems, self.contps = from_json_2d(data_dir)
        elif get_dimension() == 3:
            self.elems, self.contps = from_json_3d(data_dir)

    def from_csv(self, data_dir):
        """Load elements and contact points from csv file
        The data directory contains "element.csv" and "point.csv".

        :param data_dir: Path to data directory
        :type data_dir: str
        """
        if get_dimension() == 2:
            self.elems, self.contps = from_csv_2d(data_dir)
        elif get_dimension() == 3:
            self.elems, self.contps = from_csv_3d(data_dir)
            for k, value in self.elems.items():
                if value.vertices is None:
                    #! generate voxel vertices from center and dimension
                    # 8-point voxel https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestLinearCellDemo.png
                    # read jason file
                    import json
                    with open(str(data_dir)+"/"+value.shape_file) as f:
                        data = json.load(f)
                        d_x = float(data[0])
                        d_y = float(data[1])
                        d_z = float(data[2])
                    v_0 = [value.center[0] - d_x / 2, -
                            d_y / 2, value.center[1] - d_z / 2] #TODO: Zasto je u poslednjem clanu value.cent[1]-y koord - dz/2?
                    v_1 = [value.center[0] + d_x / 2, -
                            d_y / 2, value.center[1] - d_z / 2]
                    v_2 = [value.center[0] - d_x / 2, -
                            d_y / 2, value.center[1] + d_z / 2]
                    v_3 = [value.center[0] + d_x / 2, -
                            d_y / 2, value.center[1] + d_z / 2]
                    v_4 = [value.center[0] - d_x / 2, d_y / 2, value.center[1] - d_z / 2]
                    v_5 = [value.center[0] + d_x / 2, d_y / 2, value.center[1] - d_z / 2]
                    v_6 = [value.center[0] - d_x / 2, d_y / 2, value.center[1] + d_z / 2]
                    v_7 = [value.center[0] + d_x / 2, d_y / 2, value.center[1] + d_z / 2]
                    value.vertices = [v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7]



    def extrude_to_3d(self, extrution=1):
        set_dimension(3)

        contps = dict()
        elems = dict()
        contps_on_elements_dict = dict()
        max_nodeID = 0

        for k, value in self.contps.items():
            contps[max_nodeID] = ContPoint(max_nodeID, [value.coor[0], extrution, value.coor[1]], value.cand,
                                           value.anta, [value.tangent1[0], 0, value.tangent1[1]],
                                           [0, 1, 0], [value.normal[0], 0, value.normal[1]], value.cont_type)
            contps[max_nodeID + 1] = ContPoint(max_nodeID + 1, [value.coor[0], 0, value.coor[1]], value.cand,
                                               value.anta, [value.tangent1[0], 0, value.tangent1[1]],
                                               [0, 1, 0], [value.normal[0], 0, value.normal[1]], value.cont_type)
            if value.cand not in list(contps_on_elements_dict.keys()):
                contps_on_elements_dict[value.cand] = [
                    max_nodeID, max_nodeID + 1]
            else:
                contps_on_elements_dict[value.cand].extend(
                    [max_nodeID, max_nodeID + 1])
            max_nodeID += 2

        for k, value in self.elems.items():
            points_coord = []
            for p in contps_on_elements_dict[k]:
                points_coord.append(contps[p].coor)
            points_coord_np = np.array(points_coord)
            d_x = np.max(points_coord_np[:, 0]) - np.min(points_coord_np[:, 0])
            # d_y = np.max(points_coord_np[:, 1])-np.min(points_coord_np[:, 1])
            d_z = np.max(points_coord_np[:, 2]) - np.min(points_coord_np[:, 2])
            if d_x == 0:
                _coor_x_min = np.asarray(value.vertices)[:, 0].min()
                _coor_x_max = np.asarray(value.vertices)[:, 0].max()
            else:
                _coor_x_min = np.min(points_coord_np[:, 0])
                _coor_x_max = np.max(points_coord_np[:, 0])
            if d_z == 0:
                _coor_z_min = np.asarray(value.vertices)[:, 1].min()
                _coor_z_max = np.asarray(value.vertices)[:, 1].max()
            else:
                _coor_z_min = np.min(points_coord_np[:, 2])
                _coor_z_max = np.max(points_coord_np[:, 2])
            # mean_x = 0.5 * \
            #     np.max(points_coord_np[:, 0])+0.5*np.min(points_coord_np[:, 0])
            mean_x = value.center[0]
            # mean_y = 0.5 * \
            #     np.max(points_coord_np[:, 2])+0.5*np.min(points_coord_np[:, 2])
            mean_y = value.center[1]
            # v_0 = [np.min(points_coord_np[:, 0]), 0,
            #        np.min(points_coord_np[:, 2])]
            # v_1 = [np.max(points_coord_np[:, 0]), 0,
            #        np.min(points_coord_np[:, 2])]
            # v_2 = [np.min(points_coord_np[:, 0]), 0,
            #        np.max(points_coord_np[:, 2])]
            # v_3 = [np.max(points_coord_np[:, 0]), 0,
            #        np.max(points_coord_np[:, 2])]
            # v_4 = [np.min(points_coord_np[:, 0]), extrution,
            #        np.min(points_coord_np[:, 2])]
            # v_5 = [np.max(points_coord_np[:, 0]), extrution,
            #        np.min(points_coord_np[:, 2])]
            # v_6 = [np.min(points_coord_np[:, 0]), extrution,
            #        np.max(points_coord_np[:, 2])]
            # v_7 = [np.max(points_coord_np[:, 0]), extrution,
            #        np.max(points_coord_np[:, 2])]
            v_0 = [_coor_x_min, 0,
                   _coor_z_min]
            v_1 = [_coor_x_max, 0,
                   _coor_z_min]
            v_2 = [_coor_x_min, 0,
                   _coor_z_max]
            v_3 = [_coor_x_max, 0,
                   _coor_z_max]
            v_4 = [_coor_x_min, extrution,
                   _coor_z_min]
            v_5 = [_coor_x_max, extrution,
                   _coor_z_min]
            v_6 = [_coor_x_min, extrution,
                   _coor_z_max]
            v_7 = [_coor_x_max, extrution,
                   _coor_z_max]
            vertices = [v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7]
            elems[k] = Element(k, [mean_x, extrution / 2,
                                   mean_y], value.mass * extrution, vertices, type=value.type)
            elems[k].dl = [value.dl[0], 0, value.dl[1], 0, -value.dl[2],
                           0]  # ! attention to right-hand rotation convention
            elems[k].ll = [value.ll[0], 0, value.ll[1], 0, -value.ll[2],
                           0]  # ! attention to right-hand rotation convention
            elems[k].displacement = [0, 0, 0, 0, 0, 0]
            elems[k].mm = [[value.mass, 0, 0, 0, 0, 0], [0, value.mass, 0, 0, 0, 0], [0, 0, value.mass, 0, 0, 0],
                           [0, 0, 0, value.inerx, 0, 0], [0, 0, 0, 0, value.inerz, 0], [0, 0, 0, 0, 0, value.inery]]

            dt = 0.001
            theta = 0.7
            ### define overline mass matrix - TODO: ask for input theta and dt
            elems[k].omm = [[value.mass / (theta * dt * dt), 0, 0, 0, 0, 0],
                            [0, value.mass / (theta * dt * dt), 0, 0, 0, 0],
                            [0, 0, value.mass / (theta * dt * dt), 0, 0, 0],
                            [0, 0, 0, value.inerx / (theta * dt * dt), 0, 0],
                            [0, 0, 0, 0, value.inerz / (theta * dt * dt), 0],
                            [0, 0, 0, 0, 0, value.inery / (theta * dt * dt)]]
            if value.type == 'ground':
                print("ground extrution: ", vertices)
                print("contact points on ground", points_coord_np)
        model = Model_TH(elems, contps)
        return model


    def extrude_to_3d_z(self, extrution=1):
        set_dimension(3)

        contps = dict()
        elems = dict()
        contps_on_elements_dict = dict()
        max_nodeID = 0

        for k, value in self.contps.items():
            contps[max_nodeID] = ContPoint(max_nodeID, [value.coor[0], value.coor[1],extrution], value.cand, value.anta, [value.tangent1[0], value.tangent1[1], 0],
                                           [0, 0, 1], [value.normal[0], value.normal[1], 0], value.cont_type)
            contps[max_nodeID+1] = ContPoint(max_nodeID+1, [value.coor[0], value.coor[1], 0], value.cand, value.anta, [value.tangent1[0], value.tangent1[1], 0],
                                             [0, 0, 1], [value.normal[0], value.normal[1], 0], value.cont_type)
            if value.cand not in list(contps_on_elements_dict.keys()):
                contps_on_elements_dict[value.cand] = [
                    max_nodeID, max_nodeID+1]
            else:
                contps_on_elements_dict[value.cand].extend(
                    [max_nodeID, max_nodeID+1])
            max_nodeID += 2

        for k, value in self.elems.items():
            points_coord = []
            for p in contps_on_elements_dict[k]:
                points_coord.append(contps[p].coor)
            points_coord_np = np.array(points_coord)
            d_x = np.max(points_coord_np[:, 0])-np.min(points_coord_np[:, 0])
            d_y = np.max(points_coord_np[:, 1])-np.min(points_coord_np[:, 1])
            #d_z = np.max(points_coord_np[:, 2])-np.min(points_coord_np[:, 2])
            if d_x == 0:
                _coor_x_min = np.asarray(value.vertices)[:, 0].min()
                _coor_x_max = np.asarray(value.vertices)[:, 0].max()
            else:
                _coor_x_min = np.min(points_coord_np[:, 0])
                _coor_x_max = np.max(points_coord_np[:, 0])
            if d_y == 0:
                _coor_y_min = np.asarray(value.vertices)[:, 1].min()
                _coor_y_max = np.asarray(value.vertices)[:, 1].max()
            else:
                _coor_y_min = np.min(points_coord_np[:, 1])
                _coor_y_max = np.max(points_coord_np[:, 1])
            # mean_x = 0.5 * \
            #     np.max(points_coord_np[:, 0])+0.5*np.min(points_coord_np[:, 0])
            mean_x = value.center[0]
            # mean_y = 0.5 * \
            #     np.max(points_coord_np[:, 2])+0.5*np.min(points_coord_np[:, 2])
            mean_y = value.center[1]
            # v_0 = [np.min(points_coord_np[:, 0]), 0,
            #        np.min(points_coord_np[:, 2])]
            # v_1 = [np.max(points_coord_np[:, 0]), 0,
            #        np.min(points_coord_np[:, 2])]
            # v_2 = [np.min(points_coord_np[:, 0]), 0,
            #        np.max(points_coord_np[:, 2])]
            # v_3 = [np.max(points_coord_np[:, 0]), 0,
            #        np.max(points_coord_np[:, 2])]
            # v_4 = [np.min(points_coord_np[:, 0]), extrution,
            #        np.min(points_coord_np[:, 2])]
            # v_5 = [np.max(points_coord_np[:, 0]), extrution,
            #        np.min(points_coord_np[:, 2])]
            # v_6 = [np.min(points_coord_np[:, 0]), extrution,
            #        np.max(points_coord_np[:, 2])]
            # v_7 = [np.max(points_coord_np[:, 0]), extrution,
            #        np.max(points_coord_np[:, 2])]
            v_0 = [_coor_x_min,
                   _coor_y_min, 0]
            v_1 = [_coor_x_max,
                   _coor_y_min,0]
            v_2 = [_coor_x_min,
                   _coor_y_max, 0]
            v_3 = [_coor_x_max,
                   _coor_y_max, 0]
            v_4 = [_coor_x_min,
                   _coor_y_min, extrution]
            v_5 = [_coor_x_max,
                   _coor_y_min, extrution]
            v_6 = [_coor_x_min,
                   _coor_y_max, extrution]
            v_7 = [_coor_x_max,
                   _coor_y_max, extrution]
            vertices = [v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7]
            elems[k] = Element(k, [mean_x,
                                   mean_y, extrution/2], value.mass*extrution, vertices, type=value.type)
            elems[k].dl = [value.dl[0], value.dl[1], 0, 0, 0, value.dl[2]]
            elems[k].ll = [value.ll[0], 0, value.ll[1], 0, 0, value.ll[2]]
            elems[k].displacement = [0, 0, 0, 0, 0, 0]
            elems[k].mm = [[value.mass, 0, 0, 0, 0, 0], [0, value.mass, 0, 0, 0, 0], [0, 0, value.mass, 0, 0, 0],
                           [0, 0, 0, value.inerx, 0, 0], [0, 0, 0, 0, value.inery, 0], [0, 0, 0, 0, 0, value.inerz]]

            dt = 0.001
            theta = 0.7
            ### define overline mass matrix - TODO: ask for input theta and dt
            elems[k].omm = [[value.mass / (theta * dt * dt), 0, 0, 0, 0, 0],
                            [0, value.mass / (theta * dt * dt), 0, 0, 0, 0],
                            [0, 0, value.mass / (theta * dt * dt), 0, 0, 0],
                            [0, 0, 0, value.inerx / (theta * dt * dt), 0, 0],
                            [0, 0, 0, 0, value.inery / (theta * dt * dt), 0],
                            [0, 0, 0, 0, 0, value.inerz / (theta * dt * dt)]]
            if value.type == 'ground':
                print("ground extrution: ", vertices)
                print("contact points on ground", points_coord_np)
        model = Model_TH(elems, contps)
        return model


    def to_csv(self, data_dir):
        if get_dimension() == 2:
            self.elems, self.contps = to_csv_2d(data_dir)
        elif get_dimension() == 3:
            self.elems, self.contps = to_csv_3d(data_dir)
        pass

    def pre_check_push_over(self):
        """Check the model before push over analysis
        """

        # check counter point info
        for contact_point in self.contps.values():
            if contact_point.counterPoint is None:
                # warning
                print("Warning: contact point {} has no counter point".format(
                    contact_point.id))
                print("Recalculating counter point for all contact points......")
                self.contps = cal_anta_id(self.contps)
                break



# ### Define f for updating seismic load
    def add_gr_mot(self, value):

        for element_ID in self.elems.keys():
            if element_ID==0:
                self.elems[element_ID].sl = [0.0, 0.0, 0.0]
            else:
                acc_vect_1 = value
                acc_vect_np = np.array(acc_vect_1)
                acc_vect_np_transp = acc_vect_np.T
                #f_sl_np = mass_matrix_np.dot(acc_vect_np_transp)
                #f_sl = f_sl_np.tolist()
                self.elems[element_ID].sl = acc_vect_1
                print(f"acceleration of element{element_ID} is {acc_vect_1}")

    # ### Define f for updating seismic load in case when load is applied at one block at the bottom - shake table tests
    def add_gr_mot_sh_table(self, value):
        #     #if get_dimension() == 2:
        for element_ID in self.elems.keys():

            if value.type == "pedestal":
                # mass_matrix = self.elems[element_ID].mm
                # mass_matrix_np = np.array(mass_matrix)
                acc_vect_1 = value
                acc_vect_np = np.array(acc_vect_1)
                acc_vect_np_transp = acc_vect_np.T

                self.elems[element_ID].sl = acc_vect_1
            else:
                self.elems[element_ID].sl = [0.0, 0.0, 0.0]


# Define f for input of accelerogram 3D
    def add_gr_mot_3D(self, value):
    #     #if get_dimension() == 2:
        for element_ID in self.elems.keys():
            if value.type == "ground":
                self.elems[element_ID].sl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                #mass_matrix = self.elems[element_ID].mm
                #mass_matrix_np = np.array(mass_matrix)
                acc_vect_1 = value
                acc_vect_np = np.array(acc_vect_1)
                acc_vect_np_transp = acc_vect_np.T
                #f_sl_np = mass_matrix_np.dot(acc_vect_np_transp)
                #f_sl = f_sl_np.tolist()
                self.elems[element_ID].sl = acc_vect_1

    def add_init_vel(self):
        for element_ID in self.elems.keys():
            init_vel = [0.0, 0.0, 0.0]
            self.elems[element_ID].vel0 = init_vel


    def add_init_vel_3D(self):
        for element_ID in self.elems.keys():
            init_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.elems[element_ID].vel0 = init_vel


    def update_init_vel(self):
        for element_ID in self.elems.keys():
            self.elems[element_ID].vel0 = self.elems[element_ID].velafter


    def add_init_vel_part(self, dt):
        for element_ID in self.elems.keys():
            init_vel_part = [0.0, 0.0, 0.0]
            self.elems[element_ID].vel = init_vel_part

    def add_init_vel_part_3D(self, dt):
        for element_ID in self.elems.keys():
            init_vel_part = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.elems[element_ID].vel = init_vel_part


    def add_vel_part(self,dt):
        for element_ID in self.elems.keys():
            velocity_from_previous_step = self.elems[element_ID].velafter
            overline_mass_matrix = self.elems[element_ID].omm
            # velocity_0 = self.elems[element_ID].vel0
            vel_part1 = np.dot(overline_mass_matrix, velocity_from_previous_step)
            vel_part = np.dot(vel_part1, dt)
            self.elems[element_ID].vel = vel_part

    def add_vel_part_3D(self,dt):
        for element_ID in self.elems.keys():
            velocity_from_previous_step = self.elems[element_ID].velafter

            overline_mass_matrix = self.elems[element_ID].omm

            # velocity_0 = self.elems[element_ID].vel0
            vel_part1 = np.dot(overline_mass_matrix, velocity_from_previous_step)
            vel_part = np.dot(vel_part1, dt)
            self.elems[element_ID].vel = vel_part

    # def tot_load_step0_acc(self):
    #     for element_ID in self.elems.keys():
    #         if element_ID == 0:
    #             self.elems[element_ID].totload = [0.0, 0.0, 0.0]
    #         # else:
    #         if element_ID == 1:
    #             mass_matrix = self.elems[element_ID].mm
    #             # Gravity load
    #             f_gravity_load = [0.0, -9.81, 0.0]
    #             grav_np = np.dot(mass_matrix, f_gravity_load)
    #             grav_load = grav_np.tolist()
    #             # Seismic load
    #             seismic_load = self.elems[element_ID].sl
    #             seismic_np = np.dot(mass_matrix, seismic_load)
    #             seismic_load = seismic_np.tolist()
    #             # Sum gravity and seismic load
    #             total_load_0step_np = np.add(grav_np,seismic_np)
    #             total_load_0step = total_load_0step_np.tolist()
    #
    #             self.elems[element_ID].totload = total_load_0step
    #         if element_ID != 0 and element_ID !=1:
    #             mass_matrix = self.elems[element_ID].mm
    #             # Gravity load
    #             f_gravity_load = [0.0, -9.81, 0.0]
    #             grav_np = np.dot(mass_matrix, f_gravity_load)
    #             grav_load = grav_np.tolist()
    #             self.elems[element_ID].totload = grav_load

    def tot_load_step0_acc(self):
        for element_ID in self.elems.keys():
            #self.elems[element_ID].totload = [0.0, 0.0, 0.0]

            #if mode == 1:
                #print("acc on all elements of the arch (dl+sl)")

            if element_ID == 0:
               self.elems[element_ID].totload = [0.0, 0.0, 0.0]
            else:
                mass_matrix = self.elems[element_ID].mm
                # Gravity load
                f_gravity_load = [0.0, 9.81, 0.0]
                grav_np = np.dot(mass_matrix, f_gravity_load)
                # Seismic load
                seismic_load = self.elems[element_ID].sl
                mass_matrix = self.elems[element_ID].mm
                seismic_np = -np.dot(mass_matrix, seismic_load)
                # Sum gravity and seismic load
                total_load_np = np.add(grav_np, seismic_np)
                self.elems[element_ID].totload = total_load_np.tolist()

            # if mode == 2:
            #     print("only pedestal sl")
            #     for element_ID in self.elems.keys():
            #         if element_ID == 0 in self.elems:
            #             mass_matrix = self.elems[element_ID].mm
            #             seismic_load = self.elems[0].sl
            #             seismic_np = np.dot(mass_matrix, seismic_load)
            #             self.elems[0].totload = seismic_np.tolist()
            #             print()
            #
            # elif mode == 3:
            #     print("only pedestal dl + sl")
            #     if 0 in self.elems:
            #         mass_matrix = self.elems[0].mm
            #         # Gravity load
            #         f_gravity_load = [0.0, -9.81, 0.0]
            #         grav_np = np.dot(mass_matrix, f_gravity_load)
            #         # Seismic load
            #         seismic_load = self.elems[0].sl
            #         seismic_np = np.dot(mass_matrix, seismic_load)
            #         # Sum gravity and seismic load
            #         total_load_np = np.add(grav_np, seismic_np)
            #         self.elems[0].totload = total_load_np.tolist()
            #
            # elif mode == 4:
            #     print("on all - each arch block (dl+sl) and on pedestal dl")
            #     for element_ID in self.elems.keys():
            #         mass_matrix = self.elems[element_ID].mm
            #         # Gravity load
            #         f_gravity_load = [0.0, -9.81, 0.0]
            #         grav_np = np.dot(mass_matrix, f_gravity_load)
            #         if element_ID == 0:
            #             self.elems[element_ID].totload = grav_np.tolist()
            #         elif element_ID != 100:
            #             # Seismic load
            #             seismic_load = self.elems[element_ID].sl
            #             seismic_np = np.dot(mass_matrix, seismic_load)
            #             # Sum gravity and seismic load
            #             total_load_np = np.add(grav_np, seismic_np)
            #             self.elems[element_ID].totload = total_load_np.tolist()
            # if mode == 6:
            #     print("dl on arch and  dl+sl on pedestal")
            #     for element_ID in self.elems.keys():
            #         if element_ID == 0 in self.elems:
            #             mass_matrix = self.elems[element_ID].mm
            #             f_gravity_load = [0.0, -9.81, 0.0]
            #             grav_np = np.dot(mass_matrix, f_gravity_load)
            #             # Seismic load
            #             seismic_load = self.elems[element_ID].sl
            #             seismic_np = np.dot(mass_matrix, seismic_load)
            #             # Sum gravity and seismic load
            #             total_load_np = np.add(grav_np, seismic_np)
            #             self.elems[element_ID].totload = total_load_np.tolist()
            #         else:
            #             mass_matrix = self.elems[element_ID].mm
            #             f_gravity_load = [0.0, -9.81, 0.0]
            #             grav_np = np.dot(mass_matrix, f_gravity_load)
            #             self.elems[element_ID].totload = grav_np.tolist()
    def tot_load_step0_acc_sh_table(self):
        for element_ID in self.elems.keys():

            if element_ID == 13: #TODO: MAKE A UNIVERSAL WAY, MAYBE elems.type == 'ground':
                mass_matrix = self.elems[element_ID].mm
                # Gravity load
                f_gravity_load = [0.0, -9.81, 0.0]
                grav_np = np.dot(mass_matrix, f_gravity_load)
                # Seismic load
                seismic_load = self.elems[element_ID].sl
                seismic_np = np.dot(mass_matrix, seismic_load)
                # Sum gravity and seismic load
                total_load_np = np.add(grav_np, seismic_np)
                self.elems[element_ID].totload = total_load_np.tolist()

            else:
                mass_matrix = self.elems[element_ID].mm
                # Gravity load
                f_gravity_load = [0.0, -9.81, 0.0]
                grav_np = np.dot(mass_matrix, f_gravity_load)
                # Seismic load
                # seismic_load = self.elems[element_ID].sl
                # seismic_np = np.dot(mass_matrix, seismic_load)
                # Sum gravity and seismic load
                total_load_np = grav_np
                self.elems[element_ID].totload = total_load_np.tolist()
                # self.elems[element_ID].totload = [0.0, 0.0, 0.0]

    def tot_load_step0_acc_3D(self):
        for element_ID in self.elems.keys():
            if element_ID == 0:
                self.elems[element_ID].totload = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                mass_matrix = self.elems[element_ID].mm
                mass_matrix_np = np.array(mass_matrix)
                # Gravity load
                f_gravity_load = [0.0, -9.81,  0.0,  0.0, 0.0, 0.0]
                f_gravity_load_np = np.array(f_gravity_load)
                grav_np = np.dot(mass_matrix_np, f_gravity_load_np)
                grav_load = grav_np.tolist()
                # Seismic load
                seismic_load = self.elems[element_ID].sl
                seismic_np = np.dot(mass_matrix, seismic_load)
                seismic_load = seismic_np.tolist()
                # Sum gravity and seismic load
                total_load_0step_np = np.add(grav_np, seismic_np)
                total_load_0step = total_load_0step_np.tolist()

                self.elems[element_ID].totload = total_load_0step

    def tot_load_step0_acc_3D_z(self):
        for element_ID in self.elems.keys():
            if element_ID == 0:
                self.elems[element_ID].totload = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                mass_matrix = self.elems[element_ID].mm
                mass_matrix_np = np.array(mass_matrix)
                # Gravity load
                f_gravity_load = [0.0, -9.81,  0.0,  0.0, 0.0, 0.0]
                f_gravity_load_np = np.array(f_gravity_load)
                grav_np = np.dot(mass_matrix_np, f_gravity_load_np)
                grav_load = grav_np.tolist()
                # Seismic load
                seismic_load = self.elems[element_ID].sl
                seismic_np = np.dot(mass_matrix, seismic_load)
                seismic_load = seismic_np.tolist()
                # Sum gravity and seismic load
                total_load_0step_np = np.add(grav_np, seismic_np)
                total_load_0step = total_load_0step_np.tolist()

                self.elems[element_ID].totload = total_load_0step


    # def tot_load_steps_acc(self,dt):
    #     for element_ID in self.elems.keys():
    #         if element_ID == 0:
    #             self.elems[element_ID].totload = [0.0, 0.0, 0.0]
    #         else:
    #             #Gravity load:
    #             mass_matrix = self.elems[element_ID].mm
    #             f_gravity_load = [0.0, -9.81, 0.0]
    #             grav_np = np.dot(mass_matrix, f_gravity_load)
    #             # Seismic load
    #             seismic_load = self.elems[element_ID].sl
    #             seismic_np = np.dot(mass_matrix, seismic_load)
    #             seismic_load = seismic_np.tolist()
    #             # Sum gravity and seismic load
    #             grav_and_seism_np = np.add(grav_np, seismic_np)
    #             grav_and_seism = grav_and_seism_np.tolist()
    #
    #             # Load from velocity
    #             vel_from_previous_step=self.elems[element_ID].velafter
    #             vel_from_previous_step_np = np.array(vel_from_previous_step)
    #             overline_mass_matrix = self.elems[element_ID].omm
    #             vel_part = np.dot(overline_mass_matrix, vel_from_previous_step_np)
    #             final_vel_part = np.dot(vel_part, dt)
    #
    #             # Total load
    #             total_load_np = np.add(grav_and_seism_np, final_vel_part)
    #             total_load = total_load_np.tolist()
    #             self.elems[element_ID].totload = total_load
    def tot_load_steps_acc(self, dt):
       for element_ID in self.elems.keys():
           #print(f"Processing element_ID: {element_ID}")
           #
           # if element_ID == 100:
           #     print("Entering block for element_ID 100")
           #     self.elems[element_ID].totload = [0.0, 0.0, 0.0]

           if element_ID == 0:
               print("Entering block for element_ID 0")
               self.elems[element_ID].totload = [0.0, 0.0, 0.0]
           else:
               mass_matrix = self.elems[element_ID].mm
               f_gravity_load = [0.0, 9.81, 0.0]
               grav_np = np.dot(mass_matrix, f_gravity_load)

               # Seismic load
               seismic_load = self.elems[element_ID].sl
               seismic_np = -np.dot(mass_matrix, seismic_load)
               seismic_load = seismic_np.tolist()
               # Sum gravity and seismic load
               grav_and_seism_np = np.add(grav_np, seismic_np)
               grav_and_seism = grav_and_seism_np.tolist()

               # Load from velocity
               vel_from_previous_step = self.elems[element_ID].velafter
               vel_from_previous_step_np = np.array(vel_from_previous_step)
               overline_mass_matrix = self.elems[element_ID].omm
               vel_part = np.dot(overline_mass_matrix, vel_from_previous_step_np)
               final_vel_part = -np.dot(vel_part, dt)

               # Total load
               total_load_np = np.add(grav_and_seism_np, final_vel_part)
               total_load = total_load_np.tolist()
               self.elems[element_ID].totload = total_load
               print(f"Load of element {element_ID} is {total_load}")


    def tot_load_steps_acc_sh_table(self, dt):
       for element_ID in self.elems.keys():
           #print(f"Processing element_ID: {element_ID}")
           #
           # if element_ID == 100:
           #     print("Entering block for element_ID 100")
           #     self.elems[element_ID].totload = [0.0, 0.0, 0.0]

           if element_ID == 0:
               print("Entering block for element_ID 0")
               self.elems[element_ID].totload = [0.0, 0.0, 0.0]
           if element_ID ==13:
               mass_matrix = self.elems[element_ID].mm
               # Gravity load
               f_gravity_load = [0.0, -9.81, 0.0]
               grav_np = np.dot(mass_matrix, f_gravity_load)
               # Seismic load
               seismic_load = self.elems[element_ID].sl
               seismic_np = np.dot(mass_matrix, seismic_load)
               seismic_load = seismic_np.tolist()
               # Sum gravity and seismic load
               grav_and_seism_np = np.add(grav_np, seismic_np)
               grav_and_seism = grav_and_seism_np.tolist()

               # Load from velocity
               vel_from_previous_step = self.elems[element_ID].velafter
               vel_from_previous_step_np = np.array(vel_from_previous_step)
               overline_mass_matrix = self.elems[element_ID].omm
               vel_part = np.dot(overline_mass_matrix, vel_from_previous_step_np)
               final_vel_part = np.dot(vel_part, dt)

               # Total load
               total_load_np = np.add(grav_and_seism_np, final_vel_part)
               total_load = total_load_np.tolist()
               self.elems[element_ID].totload = total_load

           else:
               mass_matrix = self.elems[element_ID].mm
               f_gravity_load = [0.0, -9.81, 0.0]
               grav_np = np.dot(mass_matrix, f_gravity_load)

               # Load from velocity
               vel_from_previous_step = self.elems[element_ID].velafter
               vel_from_previous_step_np = np.array(vel_from_previous_step)
               overline_mass_matrix = self.elems[element_ID].omm
               vel_part = np.dot(overline_mass_matrix, vel_from_previous_step_np)
               final_vel_part = np.dot(vel_part, dt)
               total_load_np = grav_np
               # Total load
               total_load_np = np.add(grav_np, final_vel_part)
               total_load = total_load_np.tolist()
               self.elems[element_ID].totload = total_load

    def tot_load_steps_acc_OS(self, dt):
        for element_ID in self.elems.keys():
            print(f"Processing element_ID: {element_ID}")

            if element_ID == 0:
                print("Entering block for element_ID 0")
                self.elems[element_ID].totload = [0.0, 0.0, 0.0]

            else:

                print("Entering block for element_ID ..")
                # Logic for element 0: Gravity load, seismic load, and velocity load
                mass_matrix = self.elems[element_ID].mm
                f_gravity_load = [0.0, -9.81, 0.0]
                grav_np = np.dot(mass_matrix, f_gravity_load)

                # Seismic load
                seismic_load = self.elems[element_ID].sl
                seismic_np = np.dot(mass_matrix, seismic_load)

                # Sum gravity and seismic load
                grav_and_seism_np = np.add(grav_np, seismic_np)

                # Load from velocity
                vel_from_previous_step = self.elems[element_ID].velafter
                vel_from_previous_step_np = np.array(vel_from_previous_step)
                overline_mass_matrix = self.elems[element_ID].omm
                vel_part = np.dot(overline_mass_matrix, vel_from_previous_step_np)
                final_vel_part = np.dot(vel_part, dt)

                # Total load
                total_load_np = np.add(grav_and_seism_np, final_vel_part)
                total_load = total_load_np.tolist()
                self.elems[element_ID].totload = total_load


    def tot_load_steps_acc_3D(self,dt):
        for element_ID in self.elems.keys():
            if element_ID == 0:
                self.elems[element_ID].totload = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                #Gravity load:
                mass_matrix = self.elems[element_ID].mm
                f_gravity_load = [0.0, -9.81, 0.0,   0.0, 0.0, 0.0]
                grav_np = np.dot(mass_matrix, f_gravity_load)
                # Seismic load
                seismic_load = self.elems[element_ID].sl
                seismic_np = np.dot(mass_matrix, seismic_load)
                seismic_load = seismic_np.tolist()
                # Sum gravity and seismic load
                grav_and_seism_np = np.add(grav_np, seismic_np)
                grav_and_seism = grav_and_seism_np.tolist()

                # Load from velocity
                vel_from_previous_step= [self.elems[element_ID].velafter[0],self.elems[element_ID].velafter[1],0,0,0,self.elems[element_ID].velafter[5]]
                vel_from_previous_step_np = np.array(vel_from_previous_step)
                overline_mass_matrix = self.elems[element_ID].omm
                for row in range(len(overline_mass_matrix)):
                    for col in range(len(overline_mass_matrix[row])):
                        overline_mass_matrix[row][col] = float(overline_mass_matrix[row][col])
                vel_part = np.dot(overline_mass_matrix, vel_from_previous_step)
                final_vel_part = np.dot(vel_part, dt)

                # Total load
                #TODO: Impose zeroes in other directions, if the result is smaller than 10-10! Important for loads in next step
                total_load_np = np.add(grav_and_seism_np, final_vel_part)
                # for i in range(len(total_load_np)):
                #     if total_load_np[i] < abs(1e-18):
                #         total_load_np[i] = 0
                total_load = total_load_np.tolist()
                # total_load[2] = 0
                # total_load[3] = 0
                # total_load[4] = 0
                self.elems[element_ID].totload = total_load


    def total_load(self,dt):
        for element_ID in self.elems.keys():
            if element_ID == 0:
                self.elems[element_ID].totload = [0.0, 0.0, 0.0,0.0, 0.0, 0.0]
            else:
                #Gravity load:
                mass_matrix = self.elems[element_ID].mm
                f_gravity_load = [0.0, -9.81, 0.0, 0.0, 0.0, 0.0]
                grav_np = np.dot(mass_matrix, f_gravity_load)
                # Seismic load
                seismic_load = self.elems[element_ID].sl
                seismic_np = np.dot(mass_matrix, seismic_load)
                seismic_load = seismic_np.tolist()
                # Sum gravity and seismic load
                grav_and_seism_np = np.add(grav_np, seismic_np)
                grav_and_seism = grav_and_seism_np.tolist()

                # Load from velocity
                vel_from_previous_step=self.elems[element_ID].velafter
                vel_from_previous_step_np = np.array(vel_from_previous_step)
                overline_mass_matrix = self.elems[element_ID].omm
                vel_part = np.dot(overline_mass_matrix, vel_from_previous_step_np)
                final_vel_part = np.dot(vel_part, dt)

                # Total load
                total_load_np = np.add(grav_and_seism_np, final_vel_part)
                total_load = total_load_np.tolist()
                self.elems[element_ID].totload = total_load


    def calc_vel_after(self, theta, dt, index):
        theta_x_dt = 1 / (theta * dt)
        one_minus_theta = (1 - theta) / theta
        for element_ID in self.elems.keys():
            displ_el = [d for d in self.elems[element_ID].displacement]
            first_part = np.dot(displ_el, theta_x_dt) #TODO: mozda obrnuto - skalar mnozi vektor
            if index == 0:
                self.elems[element_ID].vel0 = [0.0, 0.0, 0.0]
                velocity0 = self.elems[element_ID].vel0
            else:
                self.elems[element_ID].vel0 = self.elems[element_ID].velafter
                velocity0 = self.elems[element_ID].vel0
            # if element_ID ==100:
            #     self.elems[element_ID].vel0 = [0.0, 0.0, 0.0]

            second_part = np.dot(one_minus_theta, velocity0)
            #
            velocity_after = first_part - second_part
            vel_after_list = velocity_after.tolist()
            self.elems[element_ID].velafter = vel_after_list
            print(f"Elment {element_ID}, with disp {displ_el}, has velocity after {vel_after_list}")

            # return velocity_after

    def calc_vel_after_OS(self, theta, dt, is_first_call_for_calc_vel):
        theta_x_dt = 1 / (theta * dt)
        one_minus_theta = (1 - theta) / theta
        for element_ID in self.elems.keys():
            displ_el = self.elems[element_ID].displacement
            first_part = np.dot(displ_el, theta_x_dt)

            if is_first_call_for_calc_vel:
                self.elems[element_ID].vel0 = [0.0, 0.0, 0.0]
                velocity0 = self.elems[element_ID].vel0
            else:
                self.elems[element_ID].vel0 = self.elems[element_ID].velafter
                velocity0 = self.elems[element_ID].vel0

            if element_ID == 0:
                self.elems[element_ID].vel0 = [0.0, 0.0, 0.0]

            second_part = np.dot(one_minus_theta, velocity0)

            velocity_after = first_part - second_part
            vel_after_list = velocity_after.tolist()
            self.elems[element_ID].velafter = vel_after_list
           # print()


    def calc_vel_after_3D(self, theta, dt,i):
        theta_x_dt = 1 / (theta * dt)
        one_minus_theta = (1 - theta) / theta
        for element_ID in self.elems.keys():
            displ_el = self.elems[element_ID].displacement
            first_part = np.dot(displ_el, theta_x_dt) #TODO: CHANGE
            if i == 0:
                self.elems[element_ID].vel0 = [0.0,0.0,0.0,0.0,0.0,0.0]
                velocity0 = self.elems[element_ID].vel0
            else:
                self.elems[element_ID].vel0 = self.elems[element_ID].velafter
                velocity0 = self.elems[element_ID].vel0
            second_part = np.dot(one_minus_theta, velocity0)
            #
            velocity_after = first_part - second_part
            vel_after_list = velocity_after.tolist()
            self.elems[element_ID].velafter = vel_after_list

def to_csv_2d(data_dir):
    pass


def to_csv_3d(data_dir):
    pass


def from_csv_3d(data_dir):
    contps = dict()
    elems = dict()
    # read element.csv
    df = pandas.read_csv(os.path.join(data_dir, "element.csv"))
    print("Element columns:", list(df.columns))
    for line_index, line in df.iterrows():
        # create element
        vertices = None
        elems[line['id']] = Element(line['id'], [
                                    line['cx'], line['cy'], line['cz']], line['mass'], vertices, type=line['type'], shape_file=line['shape'])
        if 'dl_fx' in df.columns:
            elems[line['id']].dl = [
                line['dl_fx'], line['dl_fy'], line['dl_fz'], line['dl_mx'], line['dl_my'], line['dl_mz']]
        if 'll_fx' in df.columns:
            elems[line['id']].ll = [
                line['ll_fx'], line['ll_fy'], line['ll_fz'], line['ll_mx'], line['ll_my'], line['ll_mz']]

        elems[line['id']].mm = [[line['mass'], 0, 0, 0, 0, 0], [0, line['mass'], 0, 0, 0, 0], [0, 0,  line['mass'], 0, 0, 0],
                                [0, 0, 0, line['mom_inert_x'], 0, 0], [0, 0, 0, 0, line['mom_inert_y'], 0], [0, 0, 0, 0, 0, line['mom_inert_z']]]


        dt = 0.001
        theta = 0.7
        ### define overline mass matrix - TODO: ask for input theta and dt
        if elems.type == 'ground':
            elems[line['id']].omm = [[line['mass'] / (theta * dt * dt), 0, 0, 0, 0, 0],
                                     [0, line['mass'] / (theta * dt * dt), 0, 0, 0, 0],
                                     [0, 0, line['mass'] / (theta * dt * dt), 0, 0, 0],
                                     [0, 0, 0, line['mom_inert_x'] / (theta * dt * dt), 0, 0],
                                     [0, 0, 0, 0, line['mom_inert_y'] / (theta * dt * dt), 0],
                                     [0, 0, 0, 0, 0, line['mom_inert_z'] / (theta * dt * dt)]]
        else:
            elems[line['id']].omm = [[line['mass'] / (theta * dt * dt), 0, 0, 0, 0, 0],
                                     [0, line['mass'] / (theta * dt * dt), 0, 0, 0, 0],
                                     [0, 0, line['mass'] / (theta * dt * dt), 0, 0, 0],
                                     [0, 0, 0, line['mom_inert_x'] / (theta * dt * dt), 0, 0],
                                     [0, 0, 0, 0, line['mom_inert_y'] / (theta * dt * dt), 0],
                                     [0, 0, 0, 0, 0, line['mom_inert_z'] / (theta * dt * dt)]]
        ### Define inverted overline mass matrix

        # elems[line['id']].invomm = [[1 / (line['mass'] / (theta * dt * dt)), 0, 0],
        #                             [0, 1 / (line['mass'] / (theta * dt * dt)), 0],
        #                             [0, 0, 1 / (line['mass'] / (theta * dt * dt))],
        #                             [0,0,0,1 / (line['mom_inert_x'] / (theta * dt * dt)),0],
        #                             [0,0,0,0,1 / (line['mom_inert_y'] / (theta * dt * dt)),0],
        #                             [0,0,0,0,0,1 / (line['mom_inert_z'] / (theta * dt * dt))]]

            # Calculate the inverse matrix
            #inv_matrix = np.linalg.inv(matrix)

            # Now, you can assign the inverse matrix to elems[..].invomm
            #elems[line['id']].invomm = inv_matrix

    # read point.csv
    df = pandas.read_csv(os.path.join(data_dir, "point.csv"))
    print("Point columns:", list(df.columns))
    for line_index, line in df.iterrows():
        if line['contact_type'] == 'friction':
            parameters = {'mu': line['mu']}
        elif line['contact_type'] == 'friction_fc_cohesion':
            parameters = {
                'mu': line['mu'], 'fc': line['fc'], 'cohesion': line['cohesion']}

        ctype = ContType(line['contact_type'], parameters)
        contps[line['id']] = ContPoint(line['id'], [line['x'], line['y'], line['z']], line['candidate_id'],
                                       line['antagonist_id'], [
                                           line['t1x'], line['t1y'], line['t1z']],
                                       [line['t2x'], line['t2y'], line['t2z']],
                                       [line['nx'], line['ny'], line['nz']], ctype)
        if 'section_height' in df.columns:
            section_h = line['section_height']
            contps[line['id']].section_h = section_h
        if 'lever' in df.columns:
            lever = line['lever']
            contps[line['id']].lever = lever
        if 'face_id' in df.columns:
            faceid = line['face_id']
            contps[line['id']].faceID = faceid
        if 'counter_point' in df.columns:
            contps[line['id']].counterPoint = line['counter_point']
    return elems, contps


def from_csv_2d(data_dir):
    contps = dict()
    elems = dict()

    # read element.csv
    df = pandas.read_csv(os.path.join(data_dir, "element.csv"))
    print("Element columns:", list(df.columns))
    for line_index, line in df.iterrows():
        # create element
        if 'shape' in df.columns:
            if line['shape'].endswith('.json'):
                with open(os.path.join(data_dir, line['shape']), 'r') as f:
                    vertices = json.load(f)
        else:
            vertices = None
        elems[line['id']] = Element(line['id'], [
                                    line['cx'], line['cy']], line['mass'], vertices, type=line['type'], shape_file=line['shape'])
        if 'dl_fx' in df.columns:
            elems[line['id']].dl = [
                line['dl_fx'], line['dl_fy'], line['dl_mz']]
        if 'll_fx' in df.columns:
            elems[line['id']].ll = [
                line['ll_fx'], line['ll_fy'], line['ll_mz']]

        ### define mass matrix
        # if 'ground' in df.columns:
        #     elems[line['id']].mm = [[0.0001, 0, 0], [0, 0.0001, 0], [0, 0, 0.0001]]
        # else:
        if 'mom_inert_x' in df.columns:
            elems[line['id']].inerx = line['mom_inert_x']
        else:
            elems[line['id']].inerx = None

        if 'mom_inert_y' in df.columns:
            elems[line['id']].inery = line['mom_inert_y']
        else:
            elems[line['id']].inery = None

        if 'mom_inert_z' in df.columns:
            elems[line['id']].inerz = line['mom_inert_z']
        else:
            elems[line['id']].inerz = None

        if line['type'] == 'ground':
            elems[line['id']].mm = [[1e13, 0, 0],
                                     [0, 1e13, 0],
                                     [0, 0, 1e13]]
        else:
            elems[line['id']].mm = [[line['mass'], 0, 0],[0,line['mass'],0],[0,0,line['mom_inert_z']]]

        ### define overline mass matrix - TODO: ask for input theta and dt
        theta = 0.7
        dt = 0.001
        if line['type'] == 'ground':
            elems[line['id']].omm = [[1e13, 0, 0],
                                     [0, 1e13, 0],
                                     [0, 0, 1e13]]

        else:
            elems[line['id']].omm = [[line['mass'] / (theta * dt * dt), 0, 0],
                                     [0, line['mass'] / (theta * dt * dt), 0],
                                     [0, 0,  line['mom_inert_z'] / (theta * dt * dt)]]


        #elems[line['id']].omm = [[line['mass']/(theta * dt * dt), 0, 0],[0,line['mass']/(theta * dt * dt),0],[0,0,line['mom_inert_z']/(theta * dt * dt)]]

        ### Define inverted overline mass matrix
        if line['type'] == 'ground':
            elems[line['id']].invomm = [[0, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 0]]
        else:
            elems[line['id']].invomm = [[1/(line['mass']/(theta * dt * dt)), 0, 0],
                                    [0,1/(line['mass']/(theta * dt * dt)),0],
                                    [0,0,1/(line['mom_inert_x']/(theta * dt * dt))]]

    # read point.csv
    df = pandas.read_csv(os.path.join(data_dir, "point.csv"))
    print("Point columns:", list(df.columns))
    for line_index, line in df.iterrows():
        if line['contact_type'] == 'friction':
            parameters = {'mu': line['mu']}
        elif line['contact_type'] == 'friction_cohesion':
            parameters = {
                'mu': line['mu'], 'cohesion': line['cohesion']}
        elif line['contact_type'] == 'friction_fc':
            parameters = {
                'mu': line['mu'], 'fc': line['fc']}
        elif line['contact_type'] == 'friction_fc_cohesion':
            parameters = {
                'mu': line['mu'], 'fc': line['fc'], 'ft': line['ft'], 'cohesion': line['cohesion']}

        ctype = ContType(line['contact_type'], parameters)
        contps[line['id']] = ContPoint(line['id'], [line['x'], line['y']], line['candidate_id'],
                                       line['antagonist_id'], [
                                           line['t1x'], line['t1y']], None, [line['nx'], line['ny']], ctype)
        if 'section_height' in df.columns:
            section_h = line['section_height']
            contps[line['id']].section_h = section_h
        if 'lever' in df.columns:
            lever = line['lever']
            contps[line['id']].lever = lever
        if 'face_id' in df.columns:
            faceid = line['face_id']
            contps[line['id']].faceID = faceid

            if 'counter_point' in df.columns:
                if line['counter_point'] != 'nan' and line['counter_point'] != 'NaN' and pandas.notnull(
                        line['counter_point']):
                    contps[line['id']].counterPoint = int(line['counter_point'])


    return elems, contps
#

def from_json_2d(data_dir):
    contps = dict()
    contpsMaxID = 0
    elems = dict()
    elemsMaxID = 0
    geo_points = []
    element_dirs = (x[0] for x in os.walk(data_dir))
    # print(element_dirs)
    for i, element in enumerate(element_dirs):
        if i == 0:
            continue
        geo_file = os.path.join(element, "geometry.txt")
        property_file = os.path.join(element, "property.txt")
        # load geometry and property file
        with open(geo_file, 'r') as f_geo:
            with open(property_file, 'r') as f_property:
                geo = json.load(f_geo)
                property = json.load(f_property)
                # iterate all the points
                for i_p, p in enumerate(geo["points"]):
                    geo_points.append(p)
                    # create node for each point
                    ctype = ContType(
                        property['contact type'], parameters=property['contact parameters'])
                    # tangent and normal direction
                    n_2d = geo["normals"][i_p]
                    n_3d = [n_2d[0], n_2d[1], 0]
                    _out_plane = np.array([0, 0, 1])
                    t_3d = np.cross(n_3d, _out_plane)
                    # create node for the point
                    contps[contpsMaxID] = ContPoint(contpsMaxID, p,
                                                    elemsMaxID, -1, [t_3d[0], t_3d[1]], None, n_2d, ctype)
                    contpsMaxID += 1
                # end iterate all the points
                # create element
                elems[elemsMaxID] = Element(
                    elemsMaxID, property['center'], property['mass'], property['vertices'], type=property['type'])
                elemsMaxID += 1
    return elems, contps


def from_json_3d(data_dir):
    """Load elements and contact points from json files in the data directory

    :param data_dir: Directory of the data.
    :type data_dir:

    """
    contps = dict()
    contpsMaxID = 0
    elems = dict()
    elemsMaxID = 0
    geo_points = []
    element_dirs = (x[0] for x in os.walk(data_dir))
    # print(element_dirs)
    for i, element in enumerate(element_dirs):
        print(element)
        if i == 0:
            continue
        geo_file = os.path.join(element, "geometry.txt")
        property_file = os.path.join(element, "property.txt")
        # load geometry and property file
        with open(geo_file, 'r') as f_geo:
            with open(property_file, 'r') as f_property:
                geo = json.load(f_geo)
                property = json.load(f_property)
                vertices = []  # a list stores the id of vertices of the current element
                # iterate all the points
                for i_p, p in enumerate(geo["points"]):
                    geo_points.append(p)
                    # create node for each point
                    ctype = ContType(
                        property['contact type'], {'mu': property['mu']})
                    # tangent and normal direction
                    n = geo["normals"][i_p]
                    # compute three orthogonal vectors
                    n = normalize(np.array(n))
                    # a vector offset to normal vector
                    help_v = n + np.ones((3), dtype=float)
                    # cross product produce a vector perpendicule to normal
                    t_1 = np.cross(n, help_v)
                    t_1 = normalize(t_1).tolist()
                    # cross product of n and t_1 produce t_2
                    t_2 = np.cross(n, t_1)
                    t_2 = normalize(t_2).tolist()
                    n = n.tolist()
                    # create node for the point
                    contps[contpsMaxID] = ContPoint(contpsMaxID, p,
                                                    elemsMaxID, -1, t_1, t_2, n, ctype)
                    vertices.append(contpsMaxID)
                    contpsMaxID += 1
                # end iterate all the points
                # create element
                elems[elemsMaxID] = Element(
                    elemsMaxID, property['center'], property['mass'], vertices)
                elemsMaxID += 1
    return elems, contps

# Define a function to call seismic load
def is_float(my_string):
    try:
        float(my_string)
        return True
    except:
        return False


def read_acc_values(csv_path):
    with open(csv_path, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader_list = list(csv_reader)

        time = []
        acc = []
        for line in csv_reader_list:
            if is_float(line[0]):
                if len(time) != 0:
                    time.append(float(line[0]) - time[len(time) - 1])
                else:
                    time.append(float(line[0]))
            else:
                time.append(0)

            acc.append(float(line[1]))
    #return acc, time
    return acc



def from_json_3d(data_dir):
    """Load elements and contact points from json files in the data directory

    :param data_dir: Directory of the data.
    :type data_dir:

    """
    contps = dict()
    contpsMaxID = 0
    elems = dict()
    elemsMaxID = 0
    geo_points = []
    element_dirs = (x[0] for x in os.walk(data_dir))
    # print(element_dirs)
    for i, element in enumerate(element_dirs):
        print(element)
        if i == 0:
            continue
        geo_file = os.path.join(element, "geometry.txt")
        property_file = os.path.join(element, "property.txt")
        # load geometry and property file
        with open(geo_file, 'r') as f_geo:
            with open(property_file, 'r') as f_property:
                geo = json.load(f_geo)
                property = json.load(f_property)
                vertices = []  # a list stores the id of vertices of the current element
                # iterate all the points
                for i_p, p in enumerate(geo["points"]):
                    geo_points.append(p)
                    # create node for each point
                    ctype = ContType(
                        property['contact type'], {'mu': property['mu']})
                    # tangent and normal direction
                    n = geo["normals"][i_p]
                    # compute three orthogonal vectors
                    n = normalize(np.array(n))
                    # a vector offset to normal vector
                    help_v = n + np.ones((3), dtype=float)
                    # cross product produce a vector perpendicule to normal
                    t_1 = np.cross(n, help_v)
                    t_1 = normalize(t_1).tolist()
                    # cross product of n and t_1 produce t_2
                    t_2 = np.cross(n, t_1)
                    t_2 = normalize(t_2).tolist()
                    n = n.tolist()
                    # create node for the point
                    contps[contpsMaxID] = ContPoint(contpsMaxID, p,
                                                    elemsMaxID, -1, t_1, t_2, n, ctype)
                    vertices.append(contpsMaxID)
                    contpsMaxID += 1
                # end iterate all the points
                # create element
                elems[elemsMaxID] = Element(
                    elemsMaxID, property['center'], property['mass'], vertices)
                elemsMaxID += 1
    return elems, contps

# # ###########################################################################################
# import pandas
# import csv
# import os
# import json
#
# import numpy as np
# from .conttype import ContType
# from ..utils.geometry import normalize
# from .contpoint import ContPoint
# from .element import Element
# from ..utils.parameter import get_dimension, set_dimension
# from ..calc.anta_id import cal_anta_id
#
# # #
# class Model_TH():
#     """A container of the model, including all i/o methods
#     """
#
#     def __init__(self, elems=dict(), contps=dict()):
#         """Constructor method
#
#         :param elems: Dictionary of elements. Key is the element id, value is Element. Defaults to empty dict()
#         :type elems: dictionary, optional
#         :param contps: Dictionary of contact points. Key is the point id, value is ContPoint. Defaults to empty dict()
#         :type contps: dictionry, optional
#         """
#         self.elems = elems
#         self.contps = contps
#
#     def from_json(self, data_dir):
#         """Load elements and contact points from json file based input
#         The data directory should be arranged as the following:
#         - data_dir
#         -- element0
#         --- geometry.txt
#         --- property.txt
#         -- element1
#         --- geometry.txt
#         --- property.txt
#         ...
#         The geometry file includes the coordinate, normal, tangent of the contact points. The property file includes the material of the element
#
#
#         :param data_dir: Path to data directory
#         :type data_dir: str
#         """
#         if get_dimension() == 2:
#             self.elems, self.contps = from_json_2d(data_dir)
#         elif get_dimension() == 3:
#             self.elems, self.contps = from_json_3d(data_dir)
#
#     def from_csv(self, data_dir):
#         """Load elements and contact points from csv file
#         The data directory contains "element.csv" and "point.csv".
#
#         :param data_dir: Path to data directory
#         :type data_dir: str
#         """
#         if get_dimension() == 2:
#             self.elems, self.contps = from_csv_2d(data_dir)
#         elif get_dimension() == 3:
#             self.elems, self.contps = from_csv_3d(data_dir)
#             for k, value in self.elems.items():
#                 if value.vertices is None:
#                     #! generate voxel vertices from center and dimension
#                     # 8-point voxel https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestLinearCellDemo.png
#                     # read jason file
#                     import json
#                     with open(str(data_dir)+"/"+value.shape_file) as f:
#                         data = json.load(f)
#                         d_x = float(data[0])
#                         d_y = float(data[1])
#                         d_z = float(data[2])
#                     v_0 = [value.center[0] - d_x / 2, -
#                             d_y / 2, value.center[1] - d_z / 2] #TODO: Zasto je u poslednjem clanu value.cent[1]-y koord - dz/2?
#                     v_1 = [value.center[0] + d_x / 2, -
#                             d_y / 2, value.center[1] - d_z / 2]
#                     v_2 = [value.center[0] - d_x / 2, -
#                             d_y / 2, value.center[1] + d_z / 2]
#                     v_3 = [value.center[0] + d_x / 2, -
#                             d_y / 2, value.center[1] + d_z / 2]
#                     v_4 = [value.center[0] - d_x / 2, d_y / 2, value.center[1] - d_z / 2]
#                     v_5 = [value.center[0] + d_x / 2, d_y / 2, value.center[1] - d_z / 2]
#                     v_6 = [value.center[0] - d_x / 2, d_y / 2, value.center[1] + d_z / 2]
#                     v_7 = [value.center[0] + d_x / 2, d_y / 2, value.center[1] + d_z / 2]
#                     value.vertices = [v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7]
#
#
#
#     def extrude_to_3d(self, extrution=1):
#         set_dimension(3)
#
#         contps = dict()
#         elems = dict()
#         contps_on_elements_dict = dict()
#         max_nodeID = 0
#
#         for k, value in self.contps.items():
#             contps[max_nodeID] = ContPoint(max_nodeID, [value.coor[0], extrution, value.coor[1]], value.cand,
#                                            value.anta, [value.tangent1[0], 0, value.tangent1[1]],
#                                            [0, 1, 0], [value.normal[0], 0, value.normal[1]], value.cont_type)
#             contps[max_nodeID + 1] = ContPoint(max_nodeID + 1, [value.coor[0], 0, value.coor[1]], value.cand,
#                                                value.anta, [value.tangent1[0], 0, value.tangent1[1]],
#                                                [0, 1, 0], [value.normal[0], 0, value.normal[1]], value.cont_type)
#             if value.cand not in list(contps_on_elements_dict.keys()):
#                 contps_on_elements_dict[value.cand] = [
#                     max_nodeID, max_nodeID + 1]
#             else:
#                 contps_on_elements_dict[value.cand].extend(
#                     [max_nodeID, max_nodeID + 1])
#             max_nodeID += 2
#
#         for k, value in self.elems.items():
#             points_coord = []
#             for p in contps_on_elements_dict[k]:
#                 points_coord.append(contps[p].coor)
#             points_coord_np = np.array(points_coord)
#             d_x = np.max(points_coord_np[:, 0]) - np.min(points_coord_np[:, 0])
#             # d_y = np.max(points_coord_np[:, 1])-np.min(points_coord_np[:, 1])
#             d_z = np.max(points_coord_np[:, 2]) - np.min(points_coord_np[:, 2])
#             if d_x == 0:
#                 _coor_x_min = np.asarray(value.vertices)[:, 0].min()
#                 _coor_x_max = np.asarray(value.vertices)[:, 0].max()
#             else:
#                 _coor_x_min = np.min(points_coord_np[:, 0])
#                 _coor_x_max = np.max(points_coord_np[:, 0])
#             if d_z == 0:
#                 _coor_z_min = np.asarray(value.vertices)[:, 1].min()
#                 _coor_z_max = np.asarray(value.vertices)[:, 1].max()
#             else:
#                 _coor_z_min = np.min(points_coord_np[:, 2])
#                 _coor_z_max = np.max(points_coord_np[:, 2])
#             # mean_x = 0.5 * \
#             #     np.max(points_coord_np[:, 0])+0.5*np.min(points_coord_np[:, 0])
#             mean_x = value.center[0]
#             # mean_y = 0.5 * \
#             #     np.max(points_coord_np[:, 2])+0.5*np.min(points_coord_np[:, 2])
#             mean_y = value.center[1]
#             # v_0 = [np.min(points_coord_np[:, 0]), 0,
#             #        np.min(points_coord_np[:, 2])]
#             # v_1 = [np.max(points_coord_np[:, 0]), 0,
#             #        np.min(points_coord_np[:, 2])]
#             # v_2 = [np.min(points_coord_np[:, 0]), 0,
#             #        np.max(points_coord_np[:, 2])]
#             # v_3 = [np.max(points_coord_np[:, 0]), 0,
#             #        np.max(points_coord_np[:, 2])]
#             # v_4 = [np.min(points_coord_np[:, 0]), extrution,
#             #        np.min(points_coord_np[:, 2])]
#             # v_5 = [np.max(points_coord_np[:, 0]), extrution,
#             #        np.min(points_coord_np[:, 2])]
#             # v_6 = [np.min(points_coord_np[:, 0]), extrution,
#             #        np.max(points_coord_np[:, 2])]
#             # v_7 = [np.max(points_coord_np[:, 0]), extrution,
#             #        np.max(points_coord_np[:, 2])]
#             v_0 = [_coor_x_min, 0,
#                    _coor_z_min]
#             v_1 = [_coor_x_max, 0,
#                    _coor_z_min]
#             v_2 = [_coor_x_min, 0,
#                    _coor_z_max]
#             v_3 = [_coor_x_max, 0,
#                    _coor_z_max]
#             v_4 = [_coor_x_min, extrution,
#                    _coor_z_min]
#             v_5 = [_coor_x_max, extrution,
#                    _coor_z_min]
#             v_6 = [_coor_x_min, extrution,
#                    _coor_z_max]
#             v_7 = [_coor_x_max, extrution,
#                    _coor_z_max]
#             vertices = [v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7]
#             elems[k] = Element(k, [mean_x, extrution / 2,
#                                    mean_y], value.mass * extrution, vertices, type=value.type)
#             elems[k].dl = [value.dl[0], 0, value.dl[1], 0, -value.dl[2],
#                            0]  # ! attention to right-hand rotation convention
#             elems[k].ll = [value.ll[0], 0, value.ll[1], 0, -value.ll[2],
#                            0]  # ! attention to right-hand rotation convention
#             elems[k].displacement = [0, 0, 0, 0, 0, 0]
#             elems[k].mm = [[value.mass, 0, 0, 0, 0, 0], [0, value.mass, 0, 0, 0, 0], [0, 0, value.mass, 0, 0, 0],
#                            [0, 0, 0, value.inerx, 0, 0], [0, 0, 0, 0, value.inerz, 0], [0, 0, 0, 0, 0, value.inery]]
#
#             dt = 0.001
#             theta = 0.7
#             ### define overline mass matrix - TODO: ask for input theta and dt
#             elems[k].omm = [[value.mass / (theta * dt * dt), 0, 0, 0, 0, 0],
#                             [0, value.mass / (theta * dt * dt), 0, 0, 0, 0],
#                             [0, 0, value.mass / (theta * dt * dt), 0, 0, 0],
#                             [0, 0, 0, value.inerx / (theta * dt * dt), 0, 0],
#                             [0, 0, 0, 0, value.inerz / (theta * dt * dt), 0],
#                             [0, 0, 0, 0, 0, value.inery / (theta * dt * dt)]]
#             if value.type == 'ground':
#                 print("ground extrution: ", vertices)
#                 print("contact points on ground", points_coord_np)
#         model = Model_TH(elems, contps)
#         return model
#
#
#     def extrude_to_3d_z(self, extrution=1):
#         set_dimension(3)
#
#         contps = dict()
#         elems = dict()
#         contps_on_elements_dict = dict()
#         max_nodeID = 0
#
#         for k, value in self.contps.items():
#             contps[max_nodeID] = ContPoint(max_nodeID, [value.coor[0], value.coor[1],extrution], value.cand, value.anta, [value.tangent1[0], value.tangent1[1], 0],
#                                            [0, 0, 1], [value.normal[0], value.normal[1], 0], value.cont_type)
#             contps[max_nodeID+1] = ContPoint(max_nodeID+1, [value.coor[0], value.coor[1], 0], value.cand, value.anta, [value.tangent1[0], value.tangent1[1], 0],
#                                              [0, 0, 1], [value.normal[0], value.normal[1], 0], value.cont_type)
#             if value.cand not in list(contps_on_elements_dict.keys()):
#                 contps_on_elements_dict[value.cand] = [
#                     max_nodeID, max_nodeID+1]
#             else:
#                 contps_on_elements_dict[value.cand].extend(
#                     [max_nodeID, max_nodeID+1])
#             max_nodeID += 2
#
#         for k, value in self.elems.items():
#             points_coord = []
#             for p in contps_on_elements_dict[k]:
#                 points_coord.append(contps[p].coor)
#             points_coord_np = np.array(points_coord)
#             d_x = np.max(points_coord_np[:, 0])-np.min(points_coord_np[:, 0])
#             d_y = np.max(points_coord_np[:, 1])-np.min(points_coord_np[:, 1])
#             #d_z = np.max(points_coord_np[:, 2])-np.min(points_coord_np[:, 2])
#             if d_x == 0:
#                 _coor_x_min = np.asarray(value.vertices)[:, 0].min()
#                 _coor_x_max = np.asarray(value.vertices)[:, 0].max()
#             else:
#                 _coor_x_min = np.min(points_coord_np[:, 0])
#                 _coor_x_max = np.max(points_coord_np[:, 0])
#             if d_y == 0:
#                 _coor_y_min = np.asarray(value.vertices)[:, 1].min()
#                 _coor_y_max = np.asarray(value.vertices)[:, 1].max()
#             else:
#                 _coor_y_min = np.min(points_coord_np[:, 1])
#                 _coor_y_max = np.max(points_coord_np[:, 1])
#             # mean_x = 0.5 * \
#             #     np.max(points_coord_np[:, 0])+0.5*np.min(points_coord_np[:, 0])
#             mean_x = value.center[0]
#             # mean_y = 0.5 * \
#             #     np.max(points_coord_np[:, 2])+0.5*np.min(points_coord_np[:, 2])
#             mean_y = value.center[1]
#             # v_0 = [np.min(points_coord_np[:, 0]), 0,
#             #        np.min(points_coord_np[:, 2])]
#             # v_1 = [np.max(points_coord_np[:, 0]), 0,
#             #        np.min(points_coord_np[:, 2])]
#             # v_2 = [np.min(points_coord_np[:, 0]), 0,
#             #        np.max(points_coord_np[:, 2])]
#             # v_3 = [np.max(points_coord_np[:, 0]), 0,
#             #        np.max(points_coord_np[:, 2])]
#             # v_4 = [np.min(points_coord_np[:, 0]), extrution,
#             #        np.min(points_coord_np[:, 2])]
#             # v_5 = [np.max(points_coord_np[:, 0]), extrution,
#             #        np.min(points_coord_np[:, 2])]
#             # v_6 = [np.min(points_coord_np[:, 0]), extrution,
#             #        np.max(points_coord_np[:, 2])]
#             # v_7 = [np.max(points_coord_np[:, 0]), extrution,
#             #        np.max(points_coord_np[:, 2])]
#             v_0 = [_coor_x_min,
#                    _coor_y_min, 0]
#             v_1 = [_coor_x_max,
#                    _coor_y_min,0]
#             v_2 = [_coor_x_min,
#                    _coor_y_max, 0]
#             v_3 = [_coor_x_max,
#                    _coor_y_max, 0]
#             v_4 = [_coor_x_min,
#                    _coor_y_min, extrution]
#             v_5 = [_coor_x_max,
#                    _coor_y_min, extrution]
#             v_6 = [_coor_x_min,
#                    _coor_y_max, extrution]
#             v_7 = [_coor_x_max,
#                    _coor_y_max, extrution]
#             vertices = [v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7]
#             elems[k] = Element(k, [mean_x,
#                                    mean_y, extrution/2], value.mass*extrution, vertices, type=value.type)
#             elems[k].dl = [value.dl[0], value.dl[1], 0, 0, 0, value.dl[2]]
#             elems[k].ll = [value.ll[0], 0, value.ll[1], 0, 0, value.ll[2]]
#             elems[k].displacement = [0, 0, 0, 0, 0, 0]
#             elems[k].mm = [[value.mass, 0, 0, 0, 0, 0], [0, value.mass, 0, 0, 0, 0], [0, 0, value.mass, 0, 0, 0],
#                            [0, 0, 0, value.inerx, 0, 0], [0, 0, 0, 0, value.inery, 0], [0, 0, 0, 0, 0, value.inerz]]
#
#             dt = 0.001
#             theta = 0.7
#             ### define overline mass matrix - TODO: ask for input theta and dt
#             elems[k].omm = [[value.mass / (theta * dt * dt), 0, 0, 0, 0, 0],
#                             [0, value.mass / (theta * dt * dt), 0, 0, 0, 0],
#                             [0, 0, value.mass / (theta * dt * dt), 0, 0, 0],
#                             [0, 0, 0, value.inerx / (theta * dt * dt), 0, 0],
#                             [0, 0, 0, 0, value.inery / (theta * dt * dt), 0],
#                             [0, 0, 0, 0, 0, value.inerz / (theta * dt * dt)]]
#             if value.type == 'ground':
#                 print("ground extrution: ", vertices)
#                 print("contact points on ground", points_coord_np)
#         model = Model_TH(elems, contps)
#         return model
#
#
#     def to_csv(self, data_dir):
#         if get_dimension() == 2:
#             self.elems, self.contps = to_csv_2d(data_dir)
#         elif get_dimension() == 3:
#             self.elems, self.contps = to_csv_3d(data_dir)
#         pass
#
#     def pre_check_push_over(self):
#         """Check the model before push over analysis
#         """
#
#         # check counter point info
#         for contact_point in self.contps.values():
#             if contact_point.counterPoint is None:
#                 # warning
#                 print("Warning: contact point {} has no counter point".format(
#                     contact_point.id))
#                 print("Recalculating counter point for all contact points......")
#                 self.contps = cal_anta_id(self.contps)
#                 break
#
#
#
#
#
# # ### Define f for updating seismic load
#     def add_gr_mot(self, value):
#     #     #if get_dimension() == 2:
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].sl = [0.0, 0.0, 0.0]
#             else:
#                 #mass_matrix = self.elems[element_ID].mm
#                 #mass_matrix_np = np.array(mass_matrix)
#                 acc_vect_1 = value
#                 acc_vect_np = np.array(acc_vect_1)
#                 acc_vect_np_transp = acc_vect_np.T
#                 #f_sl_np = mass_matrix_np.dot(acc_vect_np_transp)
#                 #f_sl = f_sl_np.tolist()
#                 self.elems[element_ID].sl = acc_vect_1
#                     #return f_sl
#             # if element_ID != 1:
#             #     f_sl_1 = [0.0, 0.0, 0.0]
#             #     self.elems[element_ID].sl = f_sl_1
#             #     #return f_sl_1
# # Define f for input of accelerogram 3D
#     def add_gr_mot_3D(self, value):
#     #     #if get_dimension() == 2:
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].sl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#             else:
#                 #mass_matrix = self.elems[element_ID].mm
#                 #mass_matrix_np = np.array(mass_matrix)
#                 acc_vect_1 = value
#                 acc_vect_np = np.array(acc_vect_1)
#                 acc_vect_np_transp = acc_vect_np.T
#                 #f_sl_np = mass_matrix_np.dot(acc_vect_np_transp)
#                 #f_sl = f_sl_np.tolist()
#                 self.elems[element_ID].sl = acc_vect_1
#
#     def add_init_vel(self):
#         for element_ID in self.elems.keys():
#             init_vel = [0.0, 0.0, 0.0]
#             self.elems[element_ID].vel0 = init_vel
#
#
#     def add_init_vel_3D(self):
#         for element_ID in self.elems.keys():
#             init_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#             self.elems[element_ID].vel0 = init_vel
#
#     # def add_init_angle(self,value):
#     #     for element_ID in self.elems.keys():
#     #         if element_ID == 0:
#     #             self.elems[element_ID].sl = [0.0, 0.0, 0.0]
#     #         else:
#     #             initial_dipsl = value
#     #             self.elems[element_ID].sl = initial_dipsl
#
#     def calc_delta_v(self):
#         for element_ID in self.elems.keys():
#             vel_after = self.elems[element_ID].velafter
#             vel_0 = self.elems[element_ID].vel0
#             subtracted = list()
#             for item1, item2 in zip(vel_after, vel_0):
#                 subtracted.append(item1 - item2)
#             #delta_vel =  subtracted
#             self.elems[element_ID].deltav = subtracted
#
#
#
#     def update_init_vel(self):
#         for element_ID in self.elems.keys():
#             self.elems[element_ID].vel0 = self.elems[element_ID].velafter
#
#
#     def add_init_vel_part(self, dt):
#         for element_ID in self.elems.keys():
#             init_vel_part = [0.0, 0.0, 0.0]
#             self.elems[element_ID].vel = init_vel_part
#
#     def add_init_vel_part_3D(self, dt):
#         for element_ID in self.elems.keys():
#             init_vel_part = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#             self.elems[element_ID].vel = init_vel_part
#
#
#     def add_vel_part(self,dt):
#         for element_ID in self.elems.keys():
#             velocity_from_previous_step = self.elems[element_ID].velafter
#             overline_mass_matrix = self.elems[element_ID].omm
#             # velocity_0 = self.elems[element_ID].vel0
#             vel_part1 = np.dot(overline_mass_matrix, velocity_from_previous_step)
#             vel_part = np.dot(vel_part1, dt)
#             self.elems[element_ID].vel = vel_part
#
#     def add_vel_part_3D(self,dt):
#         for element_ID in self.elems.keys():
#             velocity_from_previous_step = self.elems[element_ID].velafter
#
#             overline_mass_matrix = self.elems[element_ID].omm
#
#             # velocity_0 = self.elems[element_ID].vel0
#             vel_part1 = np.dot(overline_mass_matrix, velocity_from_previous_step)
#             vel_part = np.dot(vel_part1, dt)
#             self.elems[element_ID].vel = vel_part
#
#     def tot_load_step0(self):
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0]
#             else:
#                 mass_matrix = self.elems[element_ID].mm
#                 f_gravity_load = [0.0, -9.81, 0.0]
#                 grav_np = np.dot(mass_matrix, f_gravity_load)
#                 grav_load = grav_np.tolist()
#                 self.elems[element_ID].totload = grav_load
#
#     def tot_load_step0_acc(self):
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0]
#             else:
#                 mass_matrix = self.elems[element_ID].mm
#                 # Gravity load
#                 f_gravity_load = [0.0, -9.81, 0.0]
#                 grav_np = np.dot(mass_matrix, f_gravity_load)
#                 grav_load = grav_np.tolist()
#                 # Seismic load
#                 seismic_load = self.elems[element_ID].sl
#                 seismic_np = np.dot(mass_matrix, seismic_load)
#                 seismic_load = seismic_np.tolist()
#                 # Sum gravity and seismic load
#                 total_load_0step_np = np.add(grav_np,seismic_np)
#                 total_load_0step = total_load_0step_np.tolist()
#
#                 self.elems[element_ID].totload = total_load_0step
#
#
#     def tot_load_step0_acc_3D(self):
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#             else:
#                 mass_matrix = self.elems[element_ID].mm
#                 mass_matrix_np = np.array(mass_matrix)
#                 # Gravity load
#                 f_gravity_load = [0.0, -9.81,  0.0,  0.0, 0.0, 0.0]
#                 f_gravity_load_np = np.array(f_gravity_load)
#                 grav_np = np.dot(mass_matrix_np, f_gravity_load_np)
#                 grav_load = grav_np.tolist()
#                 # Seismic load
#                 seismic_load = self.elems[element_ID].sl
#                 seismic_np = np.dot(mass_matrix, seismic_load)
#                 seismic_load = seismic_np.tolist()
#                 # Sum gravity and seismic load
#                 total_load_0step_np = np.add(grav_np, seismic_np)
#                 total_load_0step = total_load_0step_np.tolist()
#
#                 self.elems[element_ID].totload = total_load_0step
#
#     def tot_load_step0_acc_3D_z(self):
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#             else:
#                 mass_matrix = self.elems[element_ID].mm
#                 mass_matrix_np = np.array(mass_matrix)
#                 # Gravity load
#                 f_gravity_load = [0.0, -9.81,  0.0,  0.0, 0.0, 0.0]
#                 f_gravity_load_np = np.array(f_gravity_load)
#                 grav_np = np.dot(mass_matrix_np, f_gravity_load_np)
#                 grav_load = grav_np.tolist()
#                 # Seismic load
#                 seismic_load = self.elems[element_ID].sl
#                 seismic_np = np.dot(mass_matrix, seismic_load)
#                 seismic_load = seismic_np.tolist()
#                 # Sum gravity and seismic load
#                 total_load_0step_np = np.add(grav_np, seismic_np)
#                 total_load_0step = total_load_0step_np.tolist()
#
#                 self.elems[element_ID].totload = total_load_0step
#
#
#     def tot_load_step1(self, initial_angle,dt,theta):
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0]
#             else:
#                 mass_matrix = self.elems[element_ID].mm
#                 f_gravity_load = [0.0, -9.81, 0.0]
#                 grav_np = np.dot(mass_matrix, f_gravity_load)
#                 init_displ_vector= [0.0, 0.0, initial_angle]
#                 theta_times_dt = theta*dt
#                 init_velocity = np.divide(init_displ_vector, theta_times_dt)
#                 f_vel = init_velocity
#                 f_vel_np = np.array(f_vel)
#                 overline_mass_matrix = self.elems[element_ID].omm
#                 vel_part = np.dot(overline_mass_matrix,f_vel_np)
#                 final_vel_part = np.dot(vel_part,dt)
#                 total_load_np = np.add(grav_np, final_vel_part)
#                 total_load = total_load_np.tolist()
#                 self.elems[element_ID].totload = total_load
#
#
#     def tot_load_stepParan(self,dt):
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0]
#             else:
#                 #Gravity load:
#                 mass_matrix = self.elems[element_ID].mm
#                 f_gravity_load = [0.0, -9.81, 0.0]
#                 grav_np = np.dot(mass_matrix, f_gravity_load)
#
#                 vel_from_previous_step=self.elems[element_ID].velafter
#                 vel_from_previous_step_np = np.array(vel_from_previous_step)
#                 overline_mass_matrix = self.elems[element_ID].omm
#                 vel_part = np.dot(overline_mass_matrix, vel_from_previous_step_np)
#                 final_vel_part = np.dot(vel_part, dt)
#                 total_load_np = np.add(grav_np, final_vel_part)
#                 total_load = total_load_np.tolist()
#                 self.elems[element_ID].totload = total_load
#
#     def tot_load_steps_acc(self,dt):
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0]
#             else:
#                 #Gravity load:
#                 mass_matrix = self.elems[element_ID].mm
#                 f_gravity_load = [0.0, -9.81, 0.0]
#                 grav_np = np.dot(mass_matrix, f_gravity_load)
#                 # Seismic load
#                 seismic_load = self.elems[element_ID].sl
#                 seismic_np = np.dot(mass_matrix, seismic_load)
#                 seismic_load = seismic_np.tolist()
#                 # Sum gravity and seismic load
#                 grav_and_seism_np = np.add(grav_np, seismic_np)
#                 grav_and_seism = grav_and_seism_np.tolist()
#
#                 # Load from velocity
#                 vel_from_previous_step=self.elems[element_ID].velafter
#                 vel_from_previous_step_np = np.array(vel_from_previous_step)
#                 overline_mass_matrix = self.elems[element_ID].omm
#                 vel_part = np.dot(overline_mass_matrix, vel_from_previous_step_np)
#                 final_vel_part = np.dot(vel_part, dt)
#
#                 # Total load
#                 total_load_np = np.add(grav_and_seism_np, final_vel_part)
#                 total_load = total_load_np.tolist()
#                 self.elems[element_ID].totload = total_load
#
#     def tot_load_steps_acc_3D(self,dt):
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#             else:
#                 #Gravity load:
#                 mass_matrix = self.elems[element_ID].mm
#                 f_gravity_load = [0.0, -9.81, 0.0,   0.0, 0.0, 0.0]
#                 grav_np = np.dot(mass_matrix, f_gravity_load)
#                 # Seismic load
#                 seismic_load = self.elems[element_ID].sl
#                 seismic_np = np.dot(mass_matrix, seismic_load)
#                 seismic_load = seismic_np.tolist()
#                 # Sum gravity and seismic load
#                 grav_and_seism_np = np.add(grav_np, seismic_np)
#                 grav_and_seism = grav_and_seism_np.tolist()
#
#                 # Load from velocity
#                 vel_from_previous_step= self.elems[element_ID].velafter
#                 vel_from_previous_step_np = np.array(vel_from_previous_step)
#                 overline_mass_matrix = self.elems[element_ID].omm
#                 for row in range(len(overline_mass_matrix)):
#                     for col in range(len(overline_mass_matrix[row])):
#                         overline_mass_matrix[row][col] = float(overline_mass_matrix[row][col])
#                 vel_part = np.dot(overline_mass_matrix, vel_from_previous_step)
#                 final_vel_part = np.dot(vel_part, dt)
#
#                 # Total load
#                 #TODO: Impose zeroes in other directions, if the result is smaller than 10-10! Important for loads in next step
#                 total_load_np = np.add(grav_and_seism_np, final_vel_part)
#                 # for i in range(len(total_load_np)):
#                 #     if total_load_np[i] < abs(1e-18):
#                 #         total_load_np[i] = 0
#                 total_load = total_load_np.tolist()
#                 # total_load[2] = 0
#                 # total_load[3] = 0
#                 # total_load[4] = 0
#                 self.elems[element_ID].totload = total_load
#
#
#     def total_load(self,dt):
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0,0.0, 0.0, 0.0]
#             else:
#                 #Gravity load:
#                 mass_matrix = self.elems[element_ID].mm
#                 f_gravity_load = [0.0, -9.81, 0.0, 0.0, 0.0, 0.0]
#                 grav_np = np.dot(mass_matrix, f_gravity_load)
#                 # Seismic load
#                 seismic_load = self.elems[element_ID].sl
#                 seismic_np = np.dot(mass_matrix, seismic_load)
#                 seismic_load = seismic_np.tolist()
#                 # Sum gravity and seismic load
#                 grav_and_seism_np = np.add(grav_np, seismic_np)
#                 grav_and_seism = grav_and_seism_np.tolist()
#
#                 # Load from velocity
#                 vel_from_previous_step=self.elems[element_ID].velafter
#                 vel_from_previous_step_np = np.array(vel_from_previous_step)
#                 overline_mass_matrix = self.elems[element_ID].omm
#                 vel_part = np.dot(overline_mass_matrix, vel_from_previous_step_np)
#                 final_vel_part = np.dot(vel_part, dt)
#
#                 # Total load
#                 total_load_np = np.add(grav_and_seism_np, final_vel_part)
#                 total_load = total_load_np.tolist()
#                 self.elems[element_ID].totload = total_load
#
#
#     def tot_load_stepNeparan(self,dt):
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0]
#             else:
#                 # Gravity load:
#                 mass_matrix = self.elems[element_ID].mm
#                 f_gravity_load = [0.0, -9.81, 0.0]
#                 grav_np = np.dot(mass_matrix, f_gravity_load)
#
#                 vel_from_previous_step = self.elems[element_ID].velafter
#                 vel_from_previous_step_np = np.array(vel_from_previous_step)
#                 overline_mass_matrix = self.elems[element_ID].omm
#                 vel_part = np.dot(overline_mass_matrix, vel_from_previous_step_np)
#                 final_vel_part = np.dot(vel_part, dt)
#                 total_load_np = np.add(grav_np, final_vel_part)
#                 total_load = total_load_np.tolist()
#                 total_load_reverse_sign = [-i for i in total_load]
#                 self.elems[element_ID].totload = total_load_reverse_sign
#
#
#
#
#
#
#     def total_load(self): #Dead load + seismic load (add gr mot +  add vel part)
#         for element_ID in self.elems.keys():
#             if element_ID == 0:
#                 self.elems[element_ID].totload = [0.0, 0.0, 0.0]
#             # #Dead load:
#             else:
#                 mass_matrix = self.elems[element_ID].mm
#                 #Load from seismic load
#                 f_sl = self.elems[element_ID].sl
#                 f_sl_np = np.array(f_sl)
#                 f_gravity_load = [0.0, -9.81, 0.0]
#                 vector_acc_and_gravity = np.add(f_sl_np, f_gravity_load)
#                 total_load_acc_and_grav_np = np.dot(mass_matrix, vector_acc_and_gravity)
#                 f_vel = self.elems[element_ID].vel
#                 f_vel_np = np.array(f_vel)
#                 total_load_np = np.add(total_load_acc_and_grav_np,f_vel_np )
#                 total_load= total_load_np.tolist()
#                 self.elems[element_ID].totload = total_load
#
#
#
#     def calc_overlinemm(self): #Dead load + seismic load (add gr mot + add inert + add vel part)
#         for element_ID, value in self.elems():
#             overline_mass_matrix = self.elems[element_ID].omm
#         return overline_mass_matrix
#
#
#     def calc_vel_after(self, theta, dt,index):
#         theta_x_dt = 1 / (theta * dt)
#         one_minus_theta = (1 - theta) / theta
#         for element_ID in self.elems.keys():
#             displ_el = self.elems[element_ID].displacement
#             first_part = np.dot(displ_el, theta_x_dt) #TODO: mozda obrnuto - skalar mnozi vektor
#             if index == 0:
#                 self.elems[element_ID].vel0 = [0.0,0.0,0.0]
#                 velocity0 = self.elems[element_ID].vel0
#             else:
#                 self.elems[element_ID].vel0 = self.elems[element_ID].velafter
#                 velocity0 = self.elems[element_ID].vel0
#             second_part = np.dot(one_minus_theta, velocity0)
#             #
#             velocity_after = first_part - second_part
#             vel_after_list = velocity_after.tolist()
#             self.elems[element_ID].velafter = vel_after_list
#             #return velocity_after
#
#
#     def calc_vel_after_3D(self, theta, dt,i):
#         theta_x_dt = 1 / (theta * dt)
#         one_minus_theta = (1 - theta) / theta
#         for element_ID in self.elems.keys():
#             displ_el = self.elems[element_ID].displacement
#             first_part = np.dot(displ_el, theta_x_dt) #TODO: CHANGE
#             if i == 0:
#                 self.elems[element_ID].vel0 = [0.0,0.0,0.0,0.0,0.0,0.0]
#                 velocity0 = self.elems[element_ID].vel0
#             else:
#                 self.elems[element_ID].vel0 = self.elems[element_ID].velafter
#                 velocity0 = self.elems[element_ID].vel0
#             second_part = np.dot(one_minus_theta, velocity0)
#             #
#             velocity_after = first_part - second_part
#             vel_after_list = velocity_after.tolist()
#             self.elems[element_ID].velafter = vel_after_list
#
# def to_csv_2d(data_dir):
#     pass
#
#
# def to_csv_3d(data_dir):
#     pass
#
#
# def from_csv_3d(data_dir):
#     contps = dict()
#     elems = dict()
#     # read element.csv
#     df = pandas.read_csv(os.path.join(data_dir, "element.csv"))
#     print("Element columns:", list(df.columns))
#     for line_index, line in df.iterrows():
#         # create element
#         vertices = None
#         elems[line['id']] = Element(line['id'], [
#                                     line['cx'], line['cy'], line['cz']], line['mass'], vertices, type=line['type'], shape_file=line['shape'])
#         if 'dl_fx' in df.columns:
#             elems[line['id']].dl = [
#                 line['dl_fx'], line['dl_fy'], line['dl_fz'], line['dl_mx'], line['dl_my'], line['dl_mz']]
#         if 'll_fx' in df.columns:
#             elems[line['id']].ll = [
#                 line['ll_fx'], line['ll_fy'], line['ll_fz'], line['ll_mx'], line['ll_my'], line['ll_mz']]
#
#         elems[line['id']].mm = [[line['mass'], 0, 0, 0, 0, 0], [0, line['mass'], 0, 0, 0, 0], [0, 0,  line['mass'], 0, 0, 0],
#                                 [0, 0, 0, line['mom_inert_x'], 0, 0], [0, 0, 0, 0, line['mom_inert_y'], 0], [0, 0, 0, 0, 0, line['mom_inert_z']]]
#
#
#         dt = 0.001
#         theta = 0.7
#         ### define overline mass matrix - TODO: ask for input theta and dt
#         if elems.type == 'ground':
#             elems[line['id']].omm = [[line['mass'] / (theta * dt * dt), 0, 0, 0, 0, 0],
#                                      [0, line['mass'] / (theta * dt * dt), 0, 0, 0, 0],
#                                      [0, 0, line['mass'] / (theta * dt * dt), 0, 0, 0],
#                                      [0, 0, 0, line['mom_inert_x'] / (theta * dt * dt), 0, 0],
#                                      [0, 0, 0, 0, line['mom_inert_y'] / (theta * dt * dt), 0],
#                                      [0, 0, 0, 0, 0, line['mom_inert_z'] / (theta * dt * dt)]]
#         else:
#             elems[line['id']].omm = [[line['mass'] / (theta * dt * dt), 0, 0, 0, 0, 0],
#                                      [0, line['mass'] / (theta * dt * dt), 0, 0, 0, 0],
#                                      [0, 0, line['mass'] / (theta * dt * dt), 0, 0, 0],
#                                      [0, 0, 0, line['mom_inert_x'] / (theta * dt * dt), 0, 0],
#                                      [0, 0, 0, 0, line['mom_inert_y'] / (theta * dt * dt), 0],
#                                      [0, 0, 0, 0, 0, line['mom_inert_z'] / (theta * dt * dt)]]
#         ### Define inverted overline mass matrix
#         elems[line['id']].invomm = [[1 / (line['mass'] / (theta * dt * dt)), 0, 0],
#                                     [0, 1 / (line['mass'] / (theta * dt * dt)), 0],
#                                     [0, 0, 1 / (line['mass'] / (theta * dt * dt))],[0,0,0,1 / (line['mom_inert_x'] / (theta * dt * dt)),0],
#                                     [0,0,0,0,1 / (line['mom_inert_y'] / (theta * dt * dt)),0], [0,0,0,0,0,1 / (line['mom_inert_z'] / (theta * dt * dt))]]
#
#
#
#     # read point.csv
#     df = pandas.read_csv(os.path.join(data_dir, "point.csv"))
#     print("Point columns:", list(df.columns))
#     for line_index, line in df.iterrows():
#         if line['contact_type'] == 'friction':
#             parameters = {'mu': line['mu']}
#         elif line['contact_type'] == 'friction_fc_cohesion':
#             parameters = {
#                 'mu': line['mu'], 'fc': line['fc'], 'cohesion': line['cohesion']}
#
#         ctype = ContType(line['contact_type'], parameters)
#         contps[line['id']] = ContPoint(line['id'], [line['x'], line['y'], line['z']], line['candidate_id'],
#                                        line['antagonist_id'], [
#                                            line['t1x'], line['t1y'], line['t1z']],
#                                        [line['t2x'], line['t2y'], line['t2z']],
#                                        [line['nx'], line['ny'], line['nz']], ctype)
#         if 'section_height' in df.columns:
#             section_h = line['section_height']
#             contps[line['id']].section_h = section_h
#         if 'lever' in df.columns:
#             lever = line['lever']
#             contps[line['id']].lever = lever
#         if 'face_id' in df.columns:
#             faceid = line['face_id']
#             contps[line['id']].faceID = faceid
#         if 'counter_point' in df.columns:
#             contps[line['id']].counterPoint = line['counter_point']
#     return elems, contps
#
#
# def from_csv_2d(data_dir):
#     contps = dict()
#     elems = dict()
#
#     # read element.csv
#     df = pandas.read_csv(os.path.join(data_dir, "element.csv"))
#     print("Element columns:", list(df.columns))
#     for line_index, line in df.iterrows():
#         # create element
#         if 'shape' in df.columns:
#             if line['shape'].endswith('.json'):
#                 with open(os.path.join(data_dir, line['shape']), 'r') as f:
#                     vertices = json.load(f)
#         else:
#             vertices = None
#         elems[line['id']] = Element(line['id'], [
#                                     line['cx'], line['cy']], line['mass'], vertices, type=line['type'], shape_file=line['shape'])
#         if 'dl_fx' in df.columns:
#             elems[line['id']].dl = [
#                 line['dl_fx'], line['dl_fy'], line['dl_mz']]
#         if 'll_fx' in df.columns:
#             elems[line['id']].ll = [
#                 line['ll_fx'], line['ll_fy'], line['ll_mz']]
#
#         ### define mass matrix
#         # if 'ground' in df.columns:
#         #     elems[line['id']].mm = [[0.0001, 0, 0], [0, 0.0001, 0], [0, 0, 0.0001]]
#         # else:
#         if 'mom_inert_x' in df.columns:
#             elems[line['id']].inerx = line['mom_inert_x']
#         else:
#             elems[line['id']].inerx = None
#
#         if 'mom_inert_y' in df.columns:
#             elems[line['id']].inery = line['mom_inert_y']
#         else:
#             elems[line['id']].inery = None
#
#         if 'mom_inert_z' in df.columns:
#             elems[line['id']].inerz = line['mom_inert_z']
#         else:
#             elems[line['id']].inerz = None
#
#         if line['type'] == 'ground':
#             elems[line['id']].mm = [[0, 0, 0],
#                                      [0, 0, 0],
#                                      [0, 0, 0]]
#         else:
#             elems[line['id']].mm = [[line['mass'], 0, 0],[0,line['mass'],0],[0,0,line['mom_inert_z']]]
#
#         ### define overline mass matrix - TODO: ask for input theta and dt
#         theta = 0.7
#         dt = 0.001
#         if line['type'] == 'ground':
#             elems[line['id']].omm = [[0, 0, 0],
#                                      [0, 0, 0],
#                                      [0, 0, 0]]
#
#         else:
#             elems[line['id']].omm = [[line['mass'] / (theta * dt * dt), 0, 0],
#                                      [0, line['mass'] / (theta * dt * dt), 0],
#                                      [0, 0,  line['mom_inert_z'] / (theta * dt * dt)]]
#
#
#         #elems[line['id']].omm = [[line['mass']/(theta * dt * dt), 0, 0],[0,line['mass']/(theta * dt * dt),0],[0,0,line['mom_inert_z']/(theta * dt * dt)]]
#
#         ### Define inverted overline mass matrix
#         if line['type'] == 'ground':
#             elems[line['id']].invomm = [[0, 0, 0],
#                                      [0, 0, 0],
#                                      [0, 0, 0]]
#         else:
#             elems[line['id']].invomm = [[1/(line['mass']/(theta * dt * dt)), 0, 0],
#                                     [0,1/(line['mass']/(theta * dt * dt)),0],
#                                     [0,0,1/(line['mom_inert_x']/(theta * dt * dt))]]
#
#     # read point.csv
#     df = pandas.read_csv(os.path.join(data_dir, "point.csv"))
#     print("Point columns:", list(df.columns))
#     for line_index, line in df.iterrows():
#         if line['contact_type'] == 'friction':
#             parameters = {'mu': line['mu']}
#         elif line['contact_type'] == 'friction_cohesion':
#             parameters = {
#                 'mu': line['mu'], 'cohesion': line['cohesion']}
#         elif line['contact_type'] == 'friction_fc':
#             parameters = {
#                 'mu': line['mu'], 'fc': line['fc']}
#         elif line['contact_type'] == 'friction_fc_cohesion':
#             parameters = {
#                 'mu': line['mu'], 'fc': line['fc'], 'ft': line['ft'], 'cohesion': line['cohesion']}
#
#         ctype = ContType(line['contact_type'], parameters)
#         contps[line['id']] = ContPoint(line['id'], [line['x'], line['y']], line['candidate_id'],
#                                        line['antagonist_id'], [
#                                            line['t1x'], line['t1y']], None, [line['nx'], line['ny']], ctype)
#         if 'section_height' in df.columns:
#             section_h = line['section_height']
#             contps[line['id']].section_h = section_h
#         if 'lever' in df.columns:
#             lever = line['lever']
#             contps[line['id']].lever = lever
#         if 'face_id' in df.columns:
#             faceid = line['face_id']
#             contps[line['id']].faceID = faceid
#             if 'counter_point' in df.columns:
#                 if line['counter_point'] != 'nan' and line['counter_point'] != 'NaN' and pandas.notnull(
#                         line['counter_point']):
#                     contps[line['id']].counterPoint = int(line['counter_point'])
#     return elems, contps
#
#
# def from_json_2d(data_dir):
#     contps = dict()
#     contpsMaxID = 0
#     elems = dict()
#     elemsMaxID = 0
#     geo_points = []
#     element_dirs = (x[0] for x in os.walk(data_dir))
#     # print(element_dirs)
#     for i, element in enumerate(element_dirs):
#         if i == 0:
#             continue
#         geo_file = os.path.join(element, "geometry.txt")
#         property_file = os.path.join(element, "property.txt")
#         # load geometry and property file
#         with open(geo_file, 'r') as f_geo:
#             with open(property_file, 'r') as f_property:
#                 geo = json.load(f_geo)
#                 property = json.load(f_property)
#                 # iterate all the points
#                 for i_p, p in enumerate(geo["points"]):
#                     geo_points.append(p)
#                     # create node for each point
#                     ctype = ContType(
#                         property['contact type'], parameters=property['contact parameters'])
#                     # tangent and normal direction
#                     n_2d = geo["normals"][i_p]
#                     n_3d = [n_2d[0], n_2d[1], 0]
#                     _out_plane = np.array([0, 0, 1])
#                     t_3d = np.cross(n_3d, _out_plane)
#                     # create node for the point
#                     contps[contpsMaxID] = ContPoint(contpsMaxID, p,
#                                                     elemsMaxID, -1, [t_3d[0], t_3d[1]], None, n_2d, ctype)
#                     contpsMaxID += 1
#                 # end iterate all the points
#                 # create element
#                 elems[elemsMaxID] = Element(
#                     elemsMaxID, property['center'], property['mass'], property['vertices'], type=property['type'])
#                 elemsMaxID += 1
#     return elems, contps
#
#
# def from_json_3d(data_dir):
#     """Load elements and contact points from json files in the data directory
#
#     :param data_dir: Directory of the data.
#     :type data_dir:
#
#     """
#     contps = dict()
#     contpsMaxID = 0
#     elems = dict()
#     elemsMaxID = 0
#     geo_points = []
#     element_dirs = (x[0] for x in os.walk(data_dir))
#     # print(element_dirs)
#     for i, element in enumerate(element_dirs):
#         print(element)
#         if i == 0:
#             continue
#         geo_file = os.path.join(element, "geometry.txt")
#         property_file = os.path.join(element, "property.txt")
#         # load geometry and property file
#         with open(geo_file, 'r') as f_geo:
#             with open(property_file, 'r') as f_property:
#                 geo = json.load(f_geo)
#                 property = json.load(f_property)
#                 vertices = []  # a list stores the id of vertices of the current element
#                 # iterate all the points
#                 for i_p, p in enumerate(geo["points"]):
#                     geo_points.append(p)
#                     # create node for each point
#                     ctype = ContType(
#                         property['contact type'], {'mu': property['mu']})
#                     # tangent and normal direction
#                     n = geo["normals"][i_p]
#                     # compute three orthogonal vectors
#                     n = normalize(np.array(n))
#                     # a vector offset to normal vector
#                     help_v = n + np.ones((3), dtype=float)
#                     # cross product produce a vector perpendicule to normal
#                     t_1 = np.cross(n, help_v)
#                     t_1 = normalize(t_1).tolist()
#                     # cross product of n and t_1 produce t_2
#                     t_2 = np.cross(n, t_1)
#                     t_2 = normalize(t_2).tolist()
#                     n = n.tolist()
#                     # create node for the point
#                     contps[contpsMaxID] = ContPoint(contpsMaxID, p,
#                                                     elemsMaxID, -1, t_1, t_2, n, ctype)
#                     vertices.append(contpsMaxID)
#                     contpsMaxID += 1
#                 # end iterate all the points
#                 # create element
#                 elems[elemsMaxID] = Element(
#                     elemsMaxID, property['center'], property['mass'], vertices)
#                 elemsMaxID += 1
#     return elems, contps
#
# # Define a function to call seismic load
# def is_float(my_string):
#     try:
#         float(my_string)
#         return True
#     except:
#         return False
#
#
# def read_acc_values(csv_path):
#     with open(csv_path, newline='') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         csv_reader_list = list(csv_reader)
#
#         time = []
#         acc = []
#         for line in csv_reader_list:
#             if is_float(line[0]):
#                 if len(time) != 0:
#                     time.append(float(line[0]) - time[len(time) - 1])
#                 else:
#                     time.append(float(line[0]))
#             else:
#                 time.append(0)
#
#             acc.append(float(line[1]))
#     #return acc, time
#     return acc
#
#
#
# def from_json_3d(data_dir):
#     """Load elements and contact points from json files in the data directory
#
#     :param data_dir: Directory of the data.
#     :type data_dir:
#
#     """
#     contps = dict()
#     contpsMaxID = 0
#     elems = dict()
#     elemsMaxID = 0
#     geo_points = []
#     element_dirs = (x[0] for x in os.walk(data_dir))
#     # print(element_dirs)
#     for i, element in enumerate(element_dirs):
#         print(element)
#         if i == 0:
#             continue
#         geo_file = os.path.join(element, "geometry.txt")
#         property_file = os.path.join(element, "property.txt")
#         # load geometry and property file
#         with open(geo_file, 'r') as f_geo:
#             with open(property_file, 'r') as f_property:
#                 geo = json.load(f_geo)
#                 property = json.load(f_property)
#                 vertices = []  # a list stores the id of vertices of the current element
#                 # iterate all the points
#                 for i_p, p in enumerate(geo["points"]):
#                     geo_points.append(p)
#                     # create node for each point
#                     ctype = ContType(
#                         property['contact type'], {'mu': property['mu']})
#                     # tangent and normal direction
#                     n = geo["normals"][i_p]
#                     # compute three orthogonal vectors
#                     n = normalize(np.array(n))
#                     # a vector offset to normal vector
#                     help_v = n + np.ones((3), dtype=float)
#                     # cross product produce a vector perpendicule to normal
#                     t_1 = np.cross(n, help_v)
#                     t_1 = normalize(t_1).tolist()
#                     # cross product of n and t_1 produce t_2
#                     t_2 = np.cross(n, t_1)
#                     t_2 = normalize(t_2).tolist()
#                     n = n.tolist()
#                     # create node for the point
#                     contps[contpsMaxID] = ContPoint(contpsMaxID, p,
#                                                     elemsMaxID, -1, t_1, t_2, n, ctype)
#                     vertices.append(contpsMaxID)
#                     contpsMaxID += 1
#                 # end iterate all the points
#                 # create element
#                 elems[elemsMaxID] = Element(
#                     elemsMaxID, property['center'], property['mass'], vertices)
#                 elemsMaxID += 1
#     return elems, contps