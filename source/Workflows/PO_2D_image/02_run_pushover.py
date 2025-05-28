import time
from datetime import datetime
import json
import random
import os
import math
import matplotlib.pyplot as plt
import pathlib
import cv2
from skimage.measure import regionprops, find_contours
from scipy import ndimage
import numpy as np
from Kinematics import *
import skimage
import tqdm
import copy
import pandas as pd
from multiprocessing import Pool
from matplotlib import collections as mc
#from elastic import solve_elastic_finitefc_associative_2d
from shapely.geometry import Polygon, LineString

plt.style.use('science')
colors = [
    "#045275",
    "#089099",
    "#7CCBA2",
    "#FCDE9C",
    "#F0746E",
    "#DC3977",
    "#7C1D6F"
]


color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(1000)]

# read from arguments
import argparse
# -i for data directory,-r for result directory
parser = argparse.ArgumentParser(description='Convert binary image to rigid block model')
parser.add_argument('-i', '--input_dir', type=str,
                    help='Directory of the input image')
parser.add_argument('-r', '--result_dir', type=str,
                    help='Directory of the result')
args = parser.parse_args()
_data_dir = args.input_dir
current_result_dir = args.result_dir
# read config
with open(_data_dir+'/config.json') as config_file:
    data_config = json.load(config_file)

# find imagename in the input directory
import glob
image_file = glob.glob(_data_dir+'/*.png')
if len(image_file) == 0:
    raise ValueError("No image file found in the input directory")
elif len(image_file) > 1:
    raise ValueError("More than one image file found in the input directory")
imagename = os.path.basename(image_file[0])

_type_name = imagename.split(".")[0]
crack_gap = data_config['crack_gap']
Estone = data_config['E_stone']
Emortar = data_config['E_mortar']
Poissonstone = data_config['Poisson_stone']
Poissonmortar = data_config['Poisson_mortar']
material_dict = {
    "E_stone": Estone,
    "E_mortar": Emortar,
    "Poisson_stone": Poissonstone,
    "Poisson_mortar": Poissonmortar,
    "wall_thickness": data_config['wall_thickness'],
    "scale_to_m_x": data_config['scale_to_m_x']}

def constrain_normal_beam(contps,elems):
    for p in contps.values():
        if elems[p.cand].type == 'beam':
            p.normal = [0,-1]
            p.tangent = [1,0]
        if elems[p.anta].type == 'beam':
            p.normal = [0,1]
            p.tangent = [1,0]

def compute_thickness(contps, elems):
    thickness_dict = {}
    for p in contps.values():
        vector_to_element_center = np.asarray(elems[p.cand].center)-np.asarray(p.coor)
        #project displacement to normal direction
        reversed_normal = -1*np.asarray(p.normal)
        normal_disp = np.dot(vector_to_element_center,reversed_normal)
        thickness = abs(normal_disp)
        
        #check if thickness is less than 0.01
        if thickness < 0.001:
            print(f"Warning: thickness of contact point {p.id} is less than 0.001m, set to 0.001m")
            thickness = 0.001
        
        thickness_dict[p.id] = thickness
    return thickness_dict



def built_neighbor_list(contps):
    #build dictinary of neighbor list for each contact point
    neighbor_list = {}
    for p in contps.values():
        p.neighbor = []
        for q in contps.values():
            dist = math.sqrt((p.coor[0]-q.coor[0])**2+(p.coor[1]-q.coor[1])**2)
            if dist<=0.01:
                p.neighbor.append(q.id)
        neighbor_list[p.id] = p.neighbor
        break
    print("Neighbor list:", neighbor_list)
    #{1: [1, 542, 2, 17599, 3, 17600, 4, 3837, 541, 2350, 553, 1358, 554, 17561, 1357, 3838, 2349, 17598]}

    return neighbor_list
        
def neighbor_list_from_csv(data_dir):
    # read point.csv
    df = pd.read_csv(os.path.join(data_dir, "point.csv"))
    neighbor_list = {}
    print("Point columns:", list(df.columns))
    for line_index, line in df.iterrows():
        neighbor_df = df[((df['x']-line['x'])**2+(df['y']-line['y'])**2)<=(0.01*756)**2]
        neighbor_list[line['id']] = neighbor_df['id'].tolist()
    #print("Neighbor list:", neighbor_list[1])
    return neighbor_list

def plot_displaced(elems,contps, factor=1, save_fig=False, show_fig=True, filename='displace_elements', control_point=[], plot_crack=True, plot_contps=True, plot_element_center=True, invert_y=False):
    """Plot displaced elements and contact points

    :param factor: Amplification of the plotted displacement, defaults to 1
    :type factor: int, optional
    """
    seismic = plt.get_cmap('seismic', 100)

    lines = []
    d = 0
    for key, value in elems.items():
        boundary_points = []
        center = value.center
        trans_x = value.displacement[0]*factor
        trans_y = value.displacement[1]*factor
        rot = value.displacement[2]*factor

        for pt in value.vertices:
            node_x = pt[0]-center[0]
            node_y = pt[1]-center[1]

            new_x = node_x*math.cos(rot)+node_y * \
                math.sin(rot)+trans_x+center[0]
            new_y = -node_x*math.sin(rot)+node_y * \
                math.cos(rot)+trans_y+center[1]
            boundary_points.append((new_x, new_y))
            # boundary_points.append((p[0], p[1]))

        for i in range(len(boundary_points)):
            lines.append([boundary_points[i-1], boundary_points[i]])
        d += 1
    lc = mc.LineCollection(lines, linewidths=0.3,color = 'black')
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    

    #! do not amplify stored displacement directly=>missing amplification in rotation
    if plot_contps:
        for k, value in contps.items():
            elem_disp = np.asarray(
                elems[value.cand].displacement)*factor
            trans_x = elem_disp[0]
            trans_y = elem_disp[1]
            rot = elem_disp[2]
            # print(f"element displacement {elem_disp}")
            elem_center = elems[value.cand].center
            # print(f"element center {elem_center}")
            node_x = value.coor[0]-elem_center[0]
            node_y = value.coor[1]-elem_center[1]
            new_x = node_x*math.cos(rot)+node_y * \
                math.sin(rot)+trans_x+elem_center[0]
            new_y = - node_x*math.sin(rot)+node_y * \
                math.cos(rot)+trans_y+elem_center[1]
            if value.gap[1] > crack_gap:
            #if value.cont_type.ft <=1e-3:
                color = 'r'
                ax.scatter(new_x,
                            new_y, c=color, marker='.', alpha=0.3,s = 2)
            else:
                # color = 'black'
                # ax.scatter(new_x,
                #             new_y, c=color, marker='.', alpha=0.3)
                pass
            
            # if value.id == control_point[0]:
            #     color = 'm'
            #     ax.scatter(new_x,
            #                 new_y, c=colors[2], marker='s', s=10)
            #     #ax.text(new_x, new_y, f"Control")
            # if value.id == control_point[1]:
            #     color = 'm'
            #     ax.scatter(new_x,
            #                 new_y, c=colors[2], marker='v', s=10)

    plt.axis('equal')
    #plt.xlim(self.xlim)
    #plt.ylim(self.ylim)
    if invert_y:
        plt.gca().invert_yaxis()
    if save_fig:
        plt.axis('off')
        plt.savefig(filename+'.png', format='png', dpi=600)
    if show_fig:
        plt.show()
    plt.close()

def _update_contp_force_2d(contps, forces):
    for i, value in enumerate(contps.values()):
        value.normal_force = forces[i*2+1]
        value.tangent_force = forces[i*2]
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
def cal_gap_2d_elastic(contps,elems,thickness_dict):
    cal_gap_2d(contps)
    for p in contps.values():
        # calculate distance from center of element to contact point
        # center = elems[p.cand].center
        # dist = math.sqrt(
        #     (p.coor[0]-center[0])**2+(p.coor[1]-center[1])**2)
        # thickness = dist
        thickness = thickness_dict[p.id]
        if elems[p.cand].type.startswith('stone'):
            E = Estone
            lamda = Poissonstone
        elif elems[contps[p.counterPoint].cand].type.startswith('stone'):
            E = Emortar
            lamda = Poissonmortar
        elif elems[p.cand].type.startswith('mortar') and elems[contps[p.counterPoint].cand].type.startswith('mortar'):
            E = Emortar
            lamda = Poissonmortar
        else:
            E = Estone
            lamda = Poissonstone

        #E = 2500e3
        #E = 1666e3
        kjn = E/thickness#approximation
        #kjn = 1e-1
        kn = kjn*p.section_h*data_config['wall_thickness']*data_config['scale_to_m_x']/4
        kt = kn/(2*(1+lamda))
        # if p.cont_type.ft <=1e-3:
        #     kt = 1e-5
        Ckt = 1/kt
        # if elems[p.cand].type.startswith('stone') or elems[contps[p.counterPoint].cand].type.startswith('stone'):
        #     Ckt = 1/(kt*50)
        t0k = p.tangent_force
        p.gap[0] = -Ckt*t0k
        #p.gap[0] = 0

def initialize_contact_force(contps):
    for p in contps.values():
        p.normal_force = 0
        p.tangent_force = 0
        p.gap = [0, 0]


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

def adjust_ft_c(contps,elems,neighbor_dict,thickness_dict,crack_strain = 5e-3,crack_gap_this_step = 1e-3):
    for p in contps.values():
        if elems[p.cand].type.startswith('ground') or elems[p.anta].type.startswith('ground'):
            continue
        if elems[p.cand].type.startswith('beam') or elems[p.anta].type.startswith('beam'):
            continue

        if p.gap[1]>crack_gap_this_step:
            p.cont_type.ft = 0
            p.cont_type.cohesion = 0
def solve_pushover(elems, contps, node_control, unit_horizontal_load, max_iteration,Aglobal,result_folder = None, data_folder=None,vertical_points = []):
    print("Push over starts:")
    forces = []
    displacements = []
    displacements_horizontal2 = []
    displacements_vertical1 = []
    displacements_vertical2 = []
    positions = []
    times = []
    initialize_contact_force(contps)
    displacement_x_beam = 0
    #neighbor_dict = built_neighbor_list(contps)
    #neighbor_dict = neighbor_list_from_csv(data_folder)
    neighbor_dict = None
    thickness_dict = compute_thickness(contps,elems)
    post_peak_start = False
    force_step = data_config['force_step']*abs(unit_horizontal_load)
    max_iterations_pushover = data_config['max_iterations_pushover']
    live_load_of_this_step_actual = 0
    i=0
    while i<=max_iterations_pushover:
        start_time = time.time()
        #live_load_of_this_step = i*unit_horizontal_load/max_iteration
        live_load_of_this_step = force_step*i
        #live_load_of_this_step = 5
        #calculate the adjusting moment
        A_matrix = cal_A_global_2d(elems, contps, sparse=True)
        if i>0:
            cal_gap_2d_elastic(contps,elems,thickness_dict)
        crack_gap_this_step = crack_gap
        adjust_ft_c(contps,elems,neighbor_dict,thickness_dict, crack_gap_this_step = crack_gap_this_step)
        if post_peak_start==False:
            print("Solving elastic-contact problem for step", i)
            for e in elems.values():
                if e.type == 'beam':
                    # e.ll = [live_load_of_this_step,
                    #         0,0]
                    e.dl[0] = live_load_of_this_step
                    # e.dl[0] = 0
                    # e.dl[1] = live_load_of_this_step
                    print("Live load live_load_of_this_step", live_load_of_this_step)
            solution = solve_elastic_finitefc_associative_2d(
                elems, contps, Aglobal=A_matrix,thickness_dict = thickness_dict,material_dict = material_dict)
            plot_displaced(elems,contps,factor=1e0, invert_y=True, save_fig=True, show_fig=False, plot_contps=True, plot_element_center=False,
                                filename=result_folder +
                                f'/pushover_elastic_step{i}_associative',
                                plot_crack=True,control_point = vertical_points)
            end_time_elastic = time.time()
        lm_factor = 1/data_config['disp_step']*np.sign(unit_horizontal_load)
        for e in elems.values():
            if e.type == 'beam':
                e.dl[0] = 0
                e.ll[0] = lm_factor                
        for p in contps.values():
            p.gap[0] = 0
            p.gap[1] = max(0,p.gap[1])#convergence issues
        print("Solving rigid-contact problem for step", i)
        solution_rigid = solve_finitefc_associative(
            elems, contps, Aglobal=A_matrix)
        plot_displaced(elems,contps,factor=1e0, invert_y=True, save_fig=True, show_fig=False, plot_contps=True, plot_element_center=False,
                                filename=result_folder +
                                f'/pushover_rigid_step{i}_associative',
                                plot_crack=True,control_point = vertical_points)
        end_time_plastic = time.time()
        # convergence check
        if post_peak_start == False:
            print(f"Step {i} in elastic regime, start convergence check")
            if solution['convergence']==False and solution_rigid['convergence']==False:
                print("Both solutions not converged, returning")
                return forces, displacements,displacements_vertical1,displacements_vertical2,positions,displacements_horizontal2,times
            elif solution['convergence']==False and solution_rigid['convergence']==True:
                print("Elastic solution not converged, rigid solution converged")
                # post peak regime
                solution = solution_rigid
                live_load_of_this_step_actual = solution_rigid['limit_force']*lm_factor
                post_peak_start = True
                times.append(end_time_plastic-start_time)
            elif solution['convergence']==True and solution_rigid['convergence']==False:
                print("Elastic solution converged, rigid solution not converged")
                # pre peak regime
                live_load_of_this_step_actual = live_load_of_this_step
                times.append(end_time_elastic-start_time)
                pass
            elif abs(solution_rigid['limit_force']*lm_factor)<=abs(live_load_of_this_step):
                print("Both solutions converged, rigid solution limit force is smaller than/equal to force from elastic solution")
                # post peak regime
                solution = solution_rigid
                live_load_of_this_step_actual = solution_rigid['limit_force']*lm_factor
                post_peak_start = True
                times.append(end_time_plastic-start_time)
            else:
                print("Both solutions converged, rigid solution limit force is larger than force from elastic solution")
                # pre peak regime
                live_load_of_this_step_actual = live_load_of_this_step
                times.append(end_time_elastic-start_time)
        else:
            print(f"Step {i} in post peak regime, start convergence check")
            solution = solution_rigid
            if solution['convergence']==False:
                print("Solution not converged, returning")
                return forces, displacements,displacements_vertical1,displacements_vertical2,positions,displacements_horizontal2,times
            live_load_of_this_step_actual = solution_rigid['limit_force']*lm_factor
            print("Load from rigid-contact solution", live_load_of_this_step_actual)
            times.append(end_time_plastic-start_time)
        _update_elem_disp_2d(contps, elems, solution["displacements"])
        forces.append(live_load_of_this_step_actual)
        displacements.append(
            contps[node_control].displacement[0])
        displacements_horizontal2.append(contps[vertical_points[1]].displacement[0])
        displacements_vertical1.append(contps[vertical_points[0]].displacement[1])
        displacements_vertical2.append(contps[vertical_points[1]].displacement[1])

        for e in elems.values():
            if e.type == 'beam':
                displacement_x_beam+=e.displacement[0]
        _update_contp_force_2d(contps, solution['contact_forces'])
        # to_csv_2d(
        #         elems, contps, f'/pushover_step{i}_associative', result_folder)
        _displace_model_2d(elems, contps)
        positions.append(contps[node_control].coor[0])

        #write results to file
        with open(result_folder+'/force_displacement.txt', 'a') as f:
            force_capacity_MPa = forces[-1]/(data_config['scale_img_to_width']*data_config['wall_thickness'])*1000
            f.write(f"{force_capacity_MPa},{displacements[-1]},{displacements_vertical1[-1]},{displacements_vertical2[-1]},{positions[-1]},{displacements_horizontal2[-1]},{times[-1]}\n")

        if live_load_of_this_step_actual <= 0 and i!=0:
            print("Live load of this step is less than or equal to 0, stopping")
            break
        
        i += 1

    return forces, displacements, displacements_vertical1, displacements_vertical2,positions,displacements_horizontal2,times

def run_one_mesh(model=None, A_matrix=None, data_folder=None, result_folder="result", boundary='double_bending', d_vertical_load=0, d_horizontal_load=0, d_moment=0, l_vertical_load=0, l_horizontal_load=0, l_moment=0, solver='associative', data_config=None,ground_y_threshold = 1e13,node_control = None,vertical_points = []):

    time_compute = 0
    load_multiplier = 0
    max_x_disp = 0
    set_dimension(2)

    model = model
    matrix_id = cv2.imread(data_folder+'/matrix_id.tiff', cv2.IMREAD_UNCHANGED)
    matrix_type = cv2.imread(
        data_folder+'/matrix_type.tiff', cv2.IMREAD_UNCHANGED)
    img_width = matrix_type.shape[1]
    img_height = matrix_type.shape[0]

    # ************************************************************** set load

    max_element_id = 0
    for e in model.elems.values():
        if 'brick_density' not in data_config.keys() or data_config['brick_density'] == None:
            density = 0
        else:
            if e.type != 'mortar':
                density = data_config['unit_density']*9.81*1e-3
            elif e.type == 'mortar':
                density = data_config['mortar_density']*9.81*1e-3

        if e.id > max_element_id:
            max_element_id = e.id
        if e.type == 'beam':
            e.dl = [d_horizontal_load*img_width,
                    d_vertical_load*data_config['scale_img_to_width']*data_config['wall_thickness']/1000, d_moment]
            e.ll = [l_horizontal_load*data_config['scale_img_to_width']*data_config['wall_thickness']/1000,
                    l_vertical_load*img_width, l_moment]#CHANGED BEFORE EVERY STEP
        else:
            e.ll = [0, 0, 0]
            e.dl = [0, density*e.mass, 0]
    matrix_id = (matrix_id*max_element_id).round().astype(np.int16)

    # ************************************************************** set material
    img_scale = 1
    if 'cohesion_scale' not in data_config.keys():
        if 'nb_contps' not in data_config.keys() or data_config['nb_contps'] == 4:
            _cohesion_scale = 0.25
        elif data_config['nb_contps'] == 2:
            _cohesion_scale = 0.5
    else:
        _cohesion_scale = data_config['cohesion_scale']
    mpa_scale = data_config['wall_thickness']
    for p in model.contps.values():
        # if model.elems[p.anta].type == 'beam':
        #     model.elems[p.cand].dl[1]+=vertical_load*p.section_h*0.5
        #     model.elems[p.cand].ll[0]+=horizontal_load*p.section_h*0.5
        img_scale = 1/p.section_h
        if model.elems[p.cand].type == 'mortar' and model.elems[p.anta].type == 'mortar':
            p.cont_type.type = 'friction_fc_cohesion'
            p.cont_type.fc = data_config['mortar_mortar_fc']*mpa_scale  # 10 times ft
            p.cont_type.ft = data_config['mortar_mortar_ft']*mpa_scale
            p.cont_type.cohesion = data_config['mortar_mortar_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_scale
            p.cont_type.mu = data_config['mortar_mortar_mu']
        elif (model.elems[p.cand].type.startswith('stone') and model.elems[p.anta].type == 'mortar') or\
                (model.elems[p.cand].type == 'mortar' and model.elems[p.anta].type.startswith('stone')):
            if 'different_vertical_horizontal_stone_mortar_interface_fc' not in data_config.keys() or bool(data_config['different_vertical_horizontal_stone_mortar_interface_fc']) == False:
                p.cont_type.type = 'friction_fc_cohesion'
                p.cont_type.fc = data_config['mortar_stone_fc']*mpa_scale
                p.cont_type.ft = data_config['mortar_stone_ft']*mpa_scale
                p.cont_type.cohesion = data_config['mortar_stone_cohesion'] * \
                    p.section_h*_cohesion_scale*mpa_scale
                p.cont_type.mu = data_config['mortar_stone_mu']
            elif bool(data_config['different_vertical_horizontal_stone_mortar_interface_fc']) == True:
                if abs(p.normal[0]) <= abs(p.normal[1]):  # horizontal face
                    p.cont_type.type = 'friction_fc_cohesion'
                    p.cont_type.fc = data_config['mortar_stone_fc']*mpa_scale
                    p.cont_type.ft = data_config['mortar_stone_ft']*mpa_scale
                    p.cont_type.cohesion = data_config['mortar_stone_cohesion'] * \
                        p.section_h*_cohesion_scale*mpa_scale
                    p.cont_type.mu = data_config['mortar_stone_mu']
                else:
                    p.cont_type.type = 'friction_fc_cohesion'
                    p.cont_type.fc = data_config['mortar_stone_fc_v']*mpa_scale
                    p.cont_type.ft = data_config['mortar_stone_ft_v']*mpa_scale
                    p.cont_type.cohesion = data_config['mortar_stone_cohesion_v'] * \
                        p.section_h*_cohesion_scale*mpa_scale
                    p.cont_type.mu = data_config['mortar_stone_mu_v']
        # elif model.elems[p.cand].type == 'ground' or model.elems[p.anta].type == 'ground' or\
        #         model.elems[p.cand].type == 'beam' or model.elems[p.anta].type == 'beam':
        elif model.elems[p.cand].type == 'ground' or model.elems[p.cand].type == 'beam':
            p.cont_type.type = 'friction_fc_cohesion'
            p.cont_type.fc = data_config['ground_beam_fc']*mpa_scale
            p.cont_type.ft = data_config['ground_beam_ft']*mpa_scale
            p.cont_type.cohesion = data_config['ground_beam_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_scale
            p.cont_type.mu = data_config['ground_beam_mu']
        elif model.elems[p.anta].type == 'ground' or model.elems[p.anta].type == 'beam':
            p.cont_type.type = 'friction_fc_cohesion'
            p.cont_type.fc = data_config['ground_beam_fc']*mpa_scale  # 10 times ft
            p.cont_type.ft = data_config['ground_beam_fc']*mpa_scale
            p.cont_type.cohesion = data_config['ground_beam_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_scale
            p.cont_type.mu = data_config['ground_beam_mu']
        
        # elif model.elems[p.cand].type == 'beam' or model.elems[p.anta].type == 'beam':
        #     p.cont_type.type = 'friction_fc_cohesion'
        #     p.cont_type.fc = data_config['mortar_stone_fc']
        #     p.cont_type.ft = data_config['mortar_stone_ft']
        #     p.cont_type.cohesion = data_config['mortar_stone_cohesion'] * \
        #         p.section_h*_cohesion_scale
        #     p.cont_type.mu = data_config['mortar_stone_mu']
        elif model.elems[p.cand].type == model.elems[p.anta].type and model.elems[p.anta].type.startswith('stone'):
            p.cont_type.type = 'friction_fc_cohesion'
            p.cont_type.fc = data_config['stone_stone_fc']*mpa_scale
            p.cont_type.ft = data_config['stone_stone_ft']*mpa_scale
            p.cont_type.cohesion = data_config['stone_stone_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_scale
            p.cont_type.mu = data_config['stone_stone_mu']
        elif model.elems[p.cand].type != model.elems[p.anta].type and model.elems[p.anta].type.startswith('stone') and model.elems[p.cand].type.startswith('stone'):
            p.cont_type.type = 'friction_fc_cohesion'
            p.cont_type.fc = data_config['stone_other_stone_fc']*mpa_scale
            p.cont_type.ft = data_config['stone_other_stone_ft']*mpa_scale
            p.cont_type.cohesion = data_config['stone_other_stone_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_scale
            p.cont_type.mu = data_config['stone_other_stone_mu']
        else:
            print(
                f"Unknow contact between {model.elems[p.cand].type} and {model.elems[p.anta].type}!")
        #apply ground interface to elements closer to the ground
        #otherwise it is very likely that the stone-mortar interface close to the ground is the first to fail
        #which in reality strong mortar is applied at this height
        # if p.coor[1] >= ground_y_threshold:
        #     p.cont_type.fc = data_config['ground_beam_fc']*mpa_scale
        #     p.cont_type.ft = data_config['ground_beam_ft']*mpa_scale
        #     p.cont_type.cohesion = data_config['ground_beam_cohesion'] * \
        #         p.section_h*_cohesion_scale*mpa_scale
        #     p.cont_type.mu = data_config['ground_beam_mu'] 
    print("number of contact points:", len(model.contps.values()))
    print("number of elments: ", len(model.elems.values()))
    # save model figure
    #write_to_csv(model.elems, model.contps, result_folder)
    # **************************************************************solve associative without crack
    # save solution(force, displacement) to txt
    with open(result_folder+'/force_displacement.txt', 'w+') as f:
        #forces, displacements, displacements_vertical1, displacements_vertical2,positions,displacements_horizontal2,times
        f.write("force_capacity_MPa,nc_dx_m,ltp_dy_m,rtp_dy_m,nc_cx_m,rtp_dx_m,time_per_step_s\n")
        
    if solver == 'associative':
        elems_1 = copy.deepcopy(model.elems)
        contps_1 = copy.deepcopy(model.contps)
        start = time.time()
        Aglobal = A_matrix
        unit_horizontal_load = l_horizontal_load
        nb_loading_steps = data_config['nb_loading_steps']
        solution = solve_pushover(
            elems_1, contps_1, node_control, unit_horizontal_load,nb_loading_steps,Aglobal=Aglobal,result_folder = result_folder,vertical_points = vertical_points,data_folder = data_folder)
        end = time.time()

        print('Associative solver time:', end - start)
        time_compute = end - start
        
        # save ka solution of step 0 
        load_multiplier = solution[0][0]
        max_x_disp = 0

       
    return {"load_multiplier": load_multiplier, "time": time_compute, "max_x_disp": max_x_disp}


# *******************************************
# ******** Configuration ********************
# *******************************************
_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)
kmodel_save_dir = current_result_dir+"/"+_type_name+"/kmodel"

hload_direction=data_config['hload_direction']
vload = data_config['vload']


_result_directory = current_result_dir+"/"+_type_name+'/run_pushover'

if os.path.exists(_result_directory+'/'):
    os.system('rm -rf '+_result_directory)
os.system('mkdir '+_result_directory)

with open(str(_result_directory)+f'/load_multiplier.txt', 'w+') as f:
    f.write(
        f'image;solver;sampleid;mm_fc;mm_ft;mm_c;mm_mu;mc_fc;mc_ft;mc_c;mc_mu;force_capacity_MPa\n')


with open(_result_directory+'/config.json', 'w') as f:
    json.dump(data_config, f, indent=4)


solver = "associative"
# ******************************************* rigid body model

def run_Vas_one_test(input):
    [imagename, sampleid, model, A_matrix,max_y,node_control,vertical_points] = input
    mm_fc = data_config['mortar_mortar_fc']
    mm_ft = data_config['mortar_mortar_ft']
    mm_c = data_config['mortar_mortar_cohesion']
    mm_mu = data_config['mortar_mortar_mu']
    mc_fc = data_config['mortar_stone_fc']
    mc_ft = data_config['mortar_stone_ft']
    mc_c = data_config['mortar_stone_cohesion']
    mc_mu = data_config['mortar_stone_mu']

    data_config_current = data_config.copy()
    # current result directory
    current_result_directory = str(
        _result_directory)+'/'+imagename.split(".")[0]+f'_sample{sampleid}'
    if os.path.exists(current_result_directory+'/'):
        os.system('rm -rf '+current_result_directory)
    os.system('mkdir '+current_result_directory)
    current_img_result_dir = current_result_directory + \
        '/'+imagename.split(".")[0]
    if os.path.exists(current_img_result_dir+'/'):
        os.system('rm -rf '+current_img_result_dir)
    os.system('mkdir '+current_img_result_dir)

    # set mesh size
    data_config_current['mortar_mortar_fc'] = mm_fc
    data_config_current['mortar_mortar_ft'] = mm_ft
    data_config_current['mortar_mortar_mu'] = mm_mu
    data_config_current['mortar_mortar_cohesion'] = mm_c
    data_config_current['mortar_stone_fc'] = mc_fc
    data_config_current['mortar_stone_ft'] = mc_ft
    data_config_current['mortar_stone_mu'] = mc_mu
    data_config_current['mortar_stone_cohesion'] = mc_c
    with open(current_result_directory+'/config.json', 'w') as f:
        json.dump(data_config_current, f, indent=4)
    
    # ******************************************* loading
    #set ground bound
    ground_y_threshold_ = max_y +data_config['ground_y_threshold_offset']
    record = run_one_mesh(model=model, A_matrix=A_matrix, data_folder=kmodel_save_dir, result_folder=current_result_directory,
                        d_vertical_load=vload, l_horizontal_load=1*hload_direction, solver=solver, boundary=data_config['boundary'], data_config=data_config_current,ground_y_threshold = ground_y_threshold_,node_control = node_control,vertical_points = vertical_points)

    # ******************************************* txt
    force_capacity_MPa = record['load_multiplier']/(data_config['scale_img_to_width']*data_config['wall_thickness'])*1000
    # write record to txt file
    with open(str(_result_directory)+f'/load_multiplier.txt', 'a') as f:
        f.write(imagename+';'+solver +
                f';{sampleid};{mm_fc};{mm_ft};{mm_c};{mm_mu};{mc_fc};{mc_ft};{mc_c};{mc_mu};{force_capacity_MPa}\n')

# get model
set_dimension(2)
model = Model()

# ************************************************************** read data
model.from_csv(kmodel_save_dir)
# scale the model to 1m
scale_x = data_config['scale_to_m_x']
scale_y = data_config['scale_to_m_y']
for e in model.elems.values():
    e.center = [e.center[0]*scale_x, e.center[1]*scale_y]
    e.mass = e.mass*scale_x*scale_y*data_config['wall_thickness']*data_config['scale_to_m_x']
    for vtc in e.vertices:
        vtc[0] *= scale_x
        vtc[1] *= scale_y
for p in model.contps.values():
    p.coor = [p.coor[0]*scale_x, p.coor[1]*scale_y]
    p.section_h *= scale_y
    p.lever *= scale_y

#get the element with maximum y
max_y = 0
min_y = 1e13
for e in model.elems.values():
    if e.center[1] > max_y:
        max_y = e.center[1]
    if e.center[1] < min_y:
        min_y = e.center[1]
#double bending condition
if data_config['boundary']== 'double_bending':
    for e in model.elems.values():
        if e.type=='beam':
            e.center = [e.center[0],(min_y+max_y)/2]
else:
    pass

A_matrix = cal_A_global_2d(model.elems, model.contps, sparse=True)

if data_config["element_id_control"]==False:
    # get the id of contp with min(x+y)
    vertical_points = [0,0]
    min_x_y = 1e13
    for p in model.contps.values():
        if model.elems[p.cand].type.startswith('mortar') or model.elems[p.cand].type.startswith('stone'):
            if p.coor[0]+p.coor[1] < min_x_y:
                min_x_y = p.coor[0]+p.coor[1]
                node_control = p.id
                vertical_points[0] = p.id
else:
    node_control = data_config['element_id_control']
# get the id fo point with max x, min y
max_x_min_y = 1e13
for p in model.contps.values():
    if model.elems[p.cand].type.startswith('mortar') or model.elems[p.cand].type.startswith('stone'):
        if -p.coor[0]+p.coor[1] < max_x_min_y:
            max_x_min_y = -p.coor[0]+p.coor[1]
            vertical_points[1] = p.id


run_Vas_one_test([imagename, 0, model, A_matrix,max_y,node_control,vertical_points])
