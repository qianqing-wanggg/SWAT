from pyevtk.hl import unstructuredGridToVTK,pointsToVTK
from pyevtk.vtk import VtkVoxel
import numpy as np
import math
from ..solve.util import rotate_3d
from copy import deepcopy

def cal_ptp_dist_tangent_project(point1,point2,normal2):
    vector_2_to_1 = np.asarray(point1)-np.asarray(point2)
    reversed_normal2 = -1*np.asarray(normal2)
    gap = np.dot(vector_2_to_1, reversed_normal2)
    gap_tangent = np.sqrt(np.linalg.norm(vector_2_to_1)**2-gap**2)
    if gap_tangent<0:
        print("Warning: gap_tangent<0")
    return gap_tangent

class VtkRenderer():
    """Renderer module to visualize 3d elements and contact points, using vtk
    """

    def __init__(self, elems, contps):
        """Constructor method

        :param elems: Elements of the model
        :type elems: dict
        :param contps: Contact points of the model
        :type contps: dict
        """
        self.elems = elems
        self.contps = contps

    def _convert_to_nodes_connectivity(self):
        """Convert elements to nodes and connectivity

        :return: nodes, connectivity
        :rtype: tuple
        """
        nodes = []  # a list of list
        node_index = 0
        connectivity = []
        element_vertices_nb = len(next(iter(self.elems.items()))[1].vertices)
        for key, value in self.elems.items():
            if element_vertices_nb != len(value.vertices):
                raise ValueError('Element have various number of vertices')
            # add nodes
            nodes.extend(value.vertices)
            # add connectivity
            connectivity.append(
                [i for i in range(node_index, node_index+len(value.vertices))])
            node_index += len(value.vertices)
        nodes_np_array = np.asarray(nodes)
        # !can only deal with model whose elements are the same type(same number of vertices)
        conne_np_array = np.asarray(connectivity)
        if len(nodes) != nodes_np_array.shape[0] or len(connectivity) != conne_np_array.shape[0] or len(nodes[0]) != nodes_np_array.shape[1] or element_vertices_nb != conne_np_array.shape[1]:
            raise Exception('Error in converting to numpy array')
        return nodes_np_array, conne_np_array

    def vtk_output(self, filename, nodes, elements, cell_type='voxel'):
        """
        Write a VTK file for the given nodes and elements.

        :param filename: Name of the output file
        :type filename: str
        :param nodes: Nodes of the model
        :type nodes: 2d array, each row is a node, each column is a coordinate x/y/z
        :param elements: Elements of the model
        :type elements: 2d array, each row is an element, each column is a node index
        :param type: Type of the cell type in vtk, defaults to 'voxel'. see https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestLinearCellDemo.png
        :type type: str, optional
        """

        if cell_type == 'voxel':
            _offsets = 8 + 8 * np.arange(elements.shape[0], dtype=np.int64)
            _cell_types = np.ones(
                elements.shape[0], dtype=np.int64) * VtkVoxel.tid
        else:
            print(cell_type)
            raise ValueError('VTK cell type not supported')

        unstructuredGridToVTK(
            filename,
            np.ascontiguousarray(nodes[:, 0]),
            np.ascontiguousarray(nodes[:, 1]),
            np.ascontiguousarray(nodes[:, 2]),
            connectivity=elements.flatten(),
            offsets=_offsets,
            cell_types=_cell_types,
            cellData=None,
            pointData=None
        )

    def plot_displaced(self, factor=1, filename='displace_elements'):
        """Plot displaced elements and contact points

        :param factor: Amplification of the plotted displacement, defaults to 1
        :type factor: int, optional
        """

        for key, value in self.elems.items():
            rot_angles = 1 * \
                np.asarray(value.displacement[3:])*factor  # !important

            center = np.asarray(value.center)
            point_coord = np.asarray(value.vertices)
            point_coord_res_center = point_coord-center

            rotated_point_coord_res_center = rotate_3d(
                point_coord_res_center, rot_angles, order='xyz')

            disp_center = np.asarray(value.displacement[:3])*factor
            new_vertices_coord = rotated_point_coord_res_center+disp_center+center

            value.vertices = new_vertices_coord.tolist()
            value.center = disp_center.tolist()

        nodes, elements = self._convert_to_nodes_connectivity()
        self.vtk_output(filename, nodes, elements, cell_type='voxel')

    def plot_displaced_points(self, factor=1,filename='displace_points'):
        """Plot displaced contact points

        :param factor: amplification factor for displacement, defaults to 1
        :type factor: int, optional
        :param filename: defaults to 'displace_points'
        :type filename: str, optional
        """
        points_coor = np.zeros((len(self.contps), 3))
        points_disp_x = np.zeros(len(self.contps))
        points_disp_y = np.zeros(len(self.contps))
        points_disp_z = np.zeros(len(self.contps))
        
        
        if factor == 0:
            points_coor = np.asarray([p.coor for p in self.contps.values()])
        else:
            point_iterator = 0
            for key, value in self.contps.items():
                
                element_of_point = self.elems[value.cand]
                rot_angles = 1 * \
                    np.asarray(element_of_point.displacement[3:])*factor  # !important

                center = np.asarray(element_of_point.center)
                point_coord = np.asarray(value.coor)
                point_coord_res_center = point_coord-center

                rotated_point_coord_res_center = rotate_3d(
                    point_coord_res_center, rot_angles, order='xyz')

                disp_center = np.asarray(element_of_point.displacement[:3])*factor
                new_vertices_coord = rotated_point_coord_res_center+disp_center+center

                points_coor[point_iterator] = new_vertices_coord
                points_disp_x[point_iterator] = new_vertices_coord[0]-point_coord[0]
                points_disp_y[point_iterator] = new_vertices_coord[1]-point_coord[1]
                points_disp_z[point_iterator] = new_vertices_coord[2]-point_coord[2]

                point_iterator += 1
        
        # get point type from cand and anta element types
        points_type = np.zeros(len(self.contps))
        point_cohesion = np.zeros(len(self.contps))
        point_damage_states_c = np.zeros(len(self.contps))
        point_damage_states_t = np.zeros(len(self.contps))
        point_damage_states_s = np.zeros(len(self.contps))
        point_normal_force = np.zeros(len(self.contps))
        point_tangent_force = np.zeros(len(self.contps))
        point_tangent_force2 = np.zeros(len(self.contps))
        point_gapn = np.zeros(len(self.contps))
        point_gapt = np.zeros(len(self.contps))
        disp_x = np.zeros(len(self.contps))
        disp_y = np.zeros(len(self.contps))
        disp_z = np.zeros(len(self.contps))
        # get normal info
        normal_x = np.zeros(len(self.contps))
        normal_y = np.zeros(len(self.contps))
        normal_z = np.zeros(len(self.contps))
        point_iterator = 0
        for _, value in self.contps.items():
            element_of_point = self.elems[value.cand]    
            element_of_anta_points = self.elems[value.anta]
            if element_of_point.type.startswith('stone') and element_of_anta_points.type.startswith('stone'):
                points_type[point_iterator] = 1
            elif (element_of_point.type.startswith('mortar') and element_of_anta_points.type.startswith('stone'))\
                or (element_of_point.type.startswith('stone') and element_of_anta_points.type.startswith('mortar')) :
                points_type[point_iterator] = 2
            elif element_of_point.type.startswith('mortar') and element_of_anta_points.type.startswith('mortar'):
                points_type[point_iterator] = 3
            elif element_of_point.type.startswith('beam') or element_of_point.type.startswith('ground')\
                or element_of_anta_points.type.startswith('beam') or element_of_anta_points.type.startswith('ground'):
                points_type[point_iterator] = 4
            else:
                points_type[point_iterator] = 5
            
            #displacement
            disp_x[point_iterator] = value.displacement[0]
            disp_y[point_iterator] = value.displacement[1]
            disp_z[point_iterator] = value.displacement[2]
            #normal
            normal_x[point_iterator] = value.normal[0]
            normal_y[point_iterator] = value.normal[1]
            normal_z[point_iterator] = value.normal[2]
            #cohesion
            point_cohesion[point_iterator] = value.cont_type.cohesion
            #damage states
            point_damage_states_c[point_iterator] = value.Dc
            point_damage_states_t[point_iterator] = value.Dt
            point_damage_states_s[point_iterator] = value.Ds
            #forces
            point_normal_force[point_iterator] = value.normal_force/value.section_h
            point_tangent_force[point_iterator] = value.tangent_force/value.section_h
            point_tangent_force2[point_iterator] = value.tangent_force2/value.section_h
            #gap
            point_gapn[point_iterator] = value.gap[2]
            point_gapt[point_iterator] = cal_ptp_dist_tangent_project(value.coor, \
                            self.contps[value.counterPoint].coor, value.normal)

            point_iterator +=1

        
        disp_data = dict()
        disp_data['disp_x'] = points_disp_x
        disp_data['disp_y'] = points_disp_y
        disp_data['disp_z'] = points_disp_z
        disp_data['type'] = points_type
        disp_data['cohesion'] = point_cohesion
        disp_data['damage_states'] = (point_damage_states_c,point_damage_states_t,point_damage_states_s)
        disp_data['contact_force'] = (point_normal_force,point_tangent_force,point_tangent_force2)
        disp_data['gap'] = (point_gapn,point_gapt,np.zeros(len(self.contps)))
        disp_data['disp'] = (disp_x,disp_y,disp_z)
        disp_data['normal'] = (normal_x,normal_y,normal_z)
        pointsToVTK(filename, np.ascontiguousarray(points_coor[:, 0]), np.ascontiguousarray(points_coor[:, 1]), \
                    np.ascontiguousarray(points_coor[:, 2]),data=disp_data)