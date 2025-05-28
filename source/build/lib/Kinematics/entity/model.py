# TODO: change from_json_3d contact parameters type
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


class Model():
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
            # for k, value in self.elems.items():
            #     if value.vertices is None:
            #         #! generate voxel vertices from center and dimension
            #         # 8-point voxel https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestLinearCellDemo.png
            #         # read jason file
            #         import json
            #         with open(str(data_dir)+"/"+value.shape_file) as f:
            #             data = json.load(f)
            #             d_x = float(data[0])
            #             d_y = float(data[1])
            #             d_z = float(data[2])
            #         v_0 = [value.center[0]-d_x/2, -
            #                d_y/2, value.center[1]-d_z/2]
            #         v_1 = [value.center[0]+d_x/2, -
            #                d_y/2, value.center[1]-d_z/2]
            #         v_2 = [value.center[0]-d_x/2, -
            #                d_y/2, value.center[1]+d_z/2]
            #         v_3 = [value.center[0]+d_x/2, -
            #                d_y/2, value.center[1]+d_z/2]
            #         v_4 = [value.center[0]-d_x/2, d_y/2, value.center[1]-d_z/2]
            #         v_5 = [value.center[0]+d_x/2, d_y/2, value.center[1]-d_z/2]
            #         v_6 = [value.center[0]-d_x/2, d_y/2, value.center[1]+d_z/2]
            #         v_7 = [value.center[0]+d_x/2, d_y/2, value.center[1]+d_z/2]
            #         value.vertices = [v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7]

    def extrude_to_3d(self, extrution=1):
        set_dimension(3)

        contps = dict()
        elems = dict()
        contps_on_elements_dict = dict()
        max_nodeID = 0

        for k, value in self.contps.items():
            contps[max_nodeID] = ContPoint(max_nodeID, [value.coor[0], extrution, value.coor[1]], value.cand, value.anta, [value.tangent1[0], 0, value.tangent1[1]],
                                           [0, 1, 0], [value.normal[0], 0, value.normal[1]], value.cont_type)
            contps[max_nodeID+1] = ContPoint(max_nodeID+1, [value.coor[0], 0, value.coor[1]], value.cand, value.anta, [value.tangent1[0], 0, value.tangent1[1]],
                                             [0, 1, 0], [value.normal[0], 0, value.normal[1]], value.cont_type)
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
            #d_y = np.max(points_coord_np[:, 1])-np.min(points_coord_np[:, 1])
            d_z = np.max(points_coord_np[:, 2])-np.min(points_coord_np[:, 2])
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
            elems[k] = Element(k, [mean_x, extrution/2,
                                   mean_y], value.mass*extrution, vertices, type=value.type)
            elems[k].dl = [value.dl[0], 0, value.dl[1],  0,-value.dl[2], 0]#! attention to right-hand rotation convention
            elems[k].ll = [value.ll[0], 0, value.ll[1],  0,-value.ll[2], 0]#! attention to right-hand rotation convention
            elems[k].displacement = [0, 0, 0, 0, 0, 0]
            if value.type == 'ground':
                print("ground extrution: ", vertices)
                print("contact points on ground", points_coord_np)
            elif value.type == 'beam':
                print("beam extrution: ", vertices)
                print("contact points on beam", points_coord_np)
        model = Model(elems, contps)
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
            if value.type == 'ground':
                print("ground extrution: ", vertices)
                print("contact points on ground", points_coord_np)
        model = Model(elems, contps)
        return model

    def to_csv(self, file_pref, data_dir):
        if get_dimension() == 2:
            to_csv_2d(self.elems, self.contps, file_pref, data_dir)
        elif get_dimension() == 3:
            to_csv_3d(self.elems, self.contps, file_pref, data_dir)

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


def to_csv_2d(elems, contps, file_pref, data_dir):
    # write result_element.csv
    # write result_contact_point.csv
    with open(data_dir+"/"+file_pref+"_element.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(('id', 'center', 'mass', 'dl', 'll',
                        'type', 'displacement', 'shape_file', 'contps'))
        for key, value in elems.items():
            # write value to one line
            writer.writerow(value.to_tuple())
    with open(data_dir+"/"+file_pref+"_contact_point.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(('id', 'coor', 'cand', 'anta', 'tangent1', 'normal', 'cont_type', 'section_h', 'lever', 'faceID',
                         'counterPoint', 'normal_force', 'tangent_force', 'mu', 'cohesion', 'ft', 'fc', 'crack_state', 'sliding_failure', 'strength_failure', 'opening_failure', 'crushing_failure', 'displacement', 'gap'))

        for key, value in contps.items():
            # write value to one line
            writer.writerow(value.to_tuple())


def to_csv_3d(elems, contps, file_pref, data_dir):
    # write result_element.csv
    # write result_contact_point.csv
    with open(data_dir+"/"+file_pref+"_element.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(('id', 'center', 'mass', 'dl', 'll',
                        'type', 'displacement', 'shape_file', 'contps'))
        for key, value in elems.items():
            # write value to one line
            writer.writerow(value.to_tuple())
    with open(data_dir+"/"+file_pref+"_contact_point.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(('id', 'coor', 'cand', 'anta', 'tangent1','tangent2', 'normal', 'cont_type', 'section_h', 'lever', 'faceID',
                         'counterPoint', 'normal_force', 'tangent_force', 'mu', 'cohesion', 'ft', 'fc', 'crack_state', 'sliding_failure', 'strength_failure', 'opening_failure', 'crushing_failure', 'displacement', 'gap'))

        for key, value in contps.items():
            # write value to one line
            writer.writerow(value.to_tuple())

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import time
def process_chunk_element_3d(chunk_df, columns):
    elems_local = dict()
    for idx, row in chunk_df.iterrows():
        vertices = None
        elem = Element(
            row['id'],
            [row['cx'], row['cy'], row['cz']],
            row['mass'],
            vertices,
            type=row['type'],
            shape_file=row['shape']
        )
        if 'dl_fx' in columns:
            elem.dl = [
                row['dl_fx'], row['dl_fy'], row['dl_fz'],
                row['dl_mx'], row['dl_my'], row['dl_mz']
            ]
        if 'll_fx' in columns:
            elem.ll = [
                row['ll_fx'], row['ll_fy'], row['ll_fz'],
                row['ll_mx'], row['ll_my'], row['ll_mz']
            ]
        elems_local[row['id']] = elem
    return elems_local

def process_chunk_element_2d(chunk_df, columns,data_dir):
    elems_local = dict()
    for idx, row in chunk_df.iterrows():
        #vertices = None
        if 'shape' in columns:
            if row['shape'].endswith('.json'):
                with open(os.path.join(data_dir, row['shape']), 'r') as f:
                    vertices = json.load(f)
        else:
            vertices = None

        elem = Element(
            row['id'],
            [row['cx'], row['cy']],
            row['mass'],
            vertices,
            type=row['type'],
            shape_file=row['shape']
        )
        if 'dl_fx' in columns:
            elem.dl = [
                row['dl_fx'], row['dl_fy'], row['dl_mz']]
        if 'll_fx' in columns:
            elem.ll = [
                row['ll_fx'], row['ll_fy'], row['ll_mz']]
        elems_local[row['id']] = elem
    return elems_local


# Helper function to process one chunk
def process_point_chunk_3d(chunk_df, columns):
    contps_local = dict()
    for idx, line in chunk_df.iterrows():
        parameters = dict()
        if line['contact_type'] == 'friction':
            parameters = {'mu': line['mu']}
        elif line['contact_type'] == 'friction_fc_cohesion':
            parameters = {
                'mu': line['mu'], 'fc': line['fc'],
                'cohesion': line['cohesion'], 'ft': line['ft']
            }
        if 'E' in columns:
            parameters['E'] = line['E']

        ctype = ContType(line['contact_type'], parameters)
        cont_point = ContPoint(
            line['id'],
            [line['x'], line['y'], line['z']],
            line['candidate_id'],
            line['antagonist_id'],
            [line['t1x'], line['t1y'], line['t1z']],
            [line['t2x'], line['t2y'], line['t2z']],
            [line['nx'], line['ny'], line['nz']],
            ctype
        )

        # Optional fields
        if 'section_area' in columns:
            cont_point.section_h = line['section_area']
        if 'lever' in columns:
            cont_point.lever = line['lever']
        if 'face_id' in columns:
            cont_point.faceID = line['face_id']
        if 'counter_point' in columns:
            cont_point.counterPoint = line['counter_point']
        if 'thickness' in columns:
            cont_point.thickness = line['thickness']

        contps_local[line['id']] = cont_point

    return contps_local

def process_point_chunk_2d(chunk_df, columns):
    contps_local = dict()
    for idx, line in chunk_df.iterrows():
        parameters = dict()
        if line['contact_type'] == 'friction':
            parameters = {'mu': line['mu']}
        elif line['contact_type'] == 'friction_fc_cohesion':
            parameters = {
                'mu': line['mu'], 'fc': line['fc'],
                'cohesion': line['cohesion'], 'ft': line['ft']
            }
        if 'E' in columns:
            parameters['E'] = line['E']

        ctype = ContType(line['contact_type'], parameters)
        cont_point = ContPoint(line['id'], [line['x'], line['y']], \
                               line['candidate_id'], line['antagonist_id'], \
                                [line['t1x'], line['t1y']], None, \
                                    [line['nx'], line['ny']], ctype)
        # Optional fields
        if 'section_height' in columns:
            cont_point.section_h = line['section_height']
        if 'lever' in columns:
            cont_point.lever = line['lever']
        if 'face_id' in columns:
            cont_point.faceID = line['face_id']
        if 'counter_point' in columns:
            cont_point.counterPoint = line['counter_point']
        if 'thickness' in columns:
            cont_point.thickness = line['thickness']

        contps_local[line['id']] = cont_point
        

    return contps_local

def from_csv_3d(data_dir):
    
    # Read CSV in chunks
    chunksize = 100000  # Tune this based on your RAM,100,000 rows = ~few hundred MB RAM only
    data_path = os.path.join(data_dir, "element.csv")

    elems = dict()

    # Read first few lines to detect columns
    df_sample = pandas.read_csv(data_path, nrows=5)

    columns = list(df_sample.columns)
    with ProcessPoolExecutor() as executor:
        futures = []
        for chunk in pandas.read_csv(data_path, chunksize=chunksize):
            future = executor.submit(process_chunk_element_3d, chunk, columns)
            #print(future.result())
            futures.append(future)
        
        # Collect results
        for future in futures:
            #print(future.result())
            result = future.result()
            elems.update(result)

    print(f"Total elements loaded: {len(elems)}")

    data_path = os.path.join(data_dir, "point.csv")

    contps = dict()

    # Detect columns first
    df_sample = pandas.read_csv(data_path, nrows=5)
    columns = list(df_sample.columns)
    print("Point columns:", columns)

    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        futures = []
        for chunk in pandas.read_csv(data_path, chunksize=chunksize):
            futures.append(executor.submit(process_point_chunk_3d, chunk, columns))
        
        # Collect results
        for future in futures:
            result = future.result()
            contps.update(result)
    end_time = time.time()
    print(f"Time taken to load contact points: {end_time - start_time} seconds")

    print(f"Total contact points loaded: {len(contps)}")
    return elems, contps


def from_csv_2d(data_dir):
    # Read CSV in chunks
    chunksize = 100000  # Tune this based on your RAM,100,000 rows = ~few hundred MB RAM only
    data_path = os.path.join(data_dir, "element.csv")

    elems = dict()

    # Read first few lines to detect columns
    df_sample = pandas.read_csv(data_path, nrows=5)

    columns = list(df_sample.columns)
    with ProcessPoolExecutor() as executor:
        futures = []
        for chunk in pandas.read_csv(data_path, chunksize=chunksize):
            future = executor.submit(process_chunk_element_2d, chunk, columns,data_dir)
            #print(future.result())
            futures.append(future)
        
        # Collect results
        for future in futures:
            #print(future.result())
            result = future.result()
            elems.update(result)

    print(f"Total elements loaded: {len(elems)}")

    data_path = os.path.join(data_dir, "point.csv")

    contps = dict()

    # Detect columns first
    df_sample = pandas.read_csv(data_path, nrows=5)
    columns = list(df_sample.columns)
    print("Point columns:", columns)

    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        futures = []
        for chunk in pandas.read_csv(data_path, chunksize=chunksize):
            futures.append(executor.submit(process_point_chunk_2d, chunk, columns))
        
        # Collect results
        for future in futures:
            result = future.result()
            contps.update(result)
    end_time = time.time()
    print(f"Time taken to load contact points: {end_time - start_time} seconds")

    print(f"Total contact points loaded: {len(contps)}")
    return elems, contps


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
