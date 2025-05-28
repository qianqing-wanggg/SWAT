"""
This script is used to run limit analysis on all models of a Vasconcelos wall
Input: 1. rigid block model of the wall, stored in "examples/result_04/"+type+"/kmodel"
        2. material combinasion samples, stored in "examples/result_04/"+type+"/samples_mortar_"+type+".csv"
        3.vertical load (0.5MPa, 0.875MPa, 1.25MPa)
        4. number of processors
Output: 1. limit load multiplier for each model, stored in "examples/result_04/"+type+"/load_multiplier.txt"
        2. plot of failure model, stored in subfolders for each model in "examples/result_04/"+type
        3. csv file storing forces and displacements on elements and contact points for each model, stored in subfolders for each model in "examples/result_04/"+type
"""
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
plt.style.use('science')
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

# find imagename in the input directory
import glob
image_file = glob.glob(_data_dir+'/*.png')
if len(image_file) == 0:
    raise ValueError("No image file found in the input directory")
elif len(image_file) > 1:
    raise ValueError("More than one image file found in the input directory")
imagename = os.path.basename(image_file[0])

#read model
_type_name = imagename.split(".")[0]
kmodel_save_dir = current_result_dir+'/'+_type_name+"/kmodel"
#read config
config_dir = _data_dir
with open(config_dir+'/config.json') as config_file:
    data_config = json.load(config_file)
#create result directory
_result_directory = current_result_dir+'/'+_type_name+'/run_limit_analysis'
if os.path.exists(_result_directory+'/'):
    os.system('rm -rf '+_result_directory)
os.system('mkdir '+_result_directory)
# scale the model to 1m
scale_x = data_config['scale_to_m_x']
scale_y = data_config['scale_to_m_y']
#set boundary
vload = data_config['vload']
hload_direction = data_config['hload_direction']
wall_thickness = data_config['wall_thickness']

def run_one_mesh(model=None, A_matrix=None, data_folder=None, result_folder="result", boundary='double_bending', d_vertical_load=0, d_horizontal_load=0, d_moment=0, l_vertical_load=0, l_horizontal_load=0, l_moment=0, solver='associative', data_config=None,ground_y_threshold = 1e13):

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
        density = 0
        if e.id > max_element_id:
            max_element_id = e.id
        lm_factor = l_horizontal_load*img_width*wall_thickness/1000
        if e.type == 'beam':
            e.dl = [0,
                    d_vertical_load*img_width*wall_thickness/1000, d_moment]
            e.ll = [lm_factor,
                    0, l_moment]
            # if boundary == 'double_bending':
            #     e.center = [img_width/2, img_height/2]
            # elif boundary == 'zero_moment_uppder_surf':
            #     e.center = [img_width/2, img_height/2-0.5]
            # else:
            #     e.center = [img_width/2, 2]
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
    mpa_to_2d = wall_thickness
    for p in model.contps.values():
        # if model.elems[p.anta].type == 'beam':
        #     model.elems[p.cand].dl[1]+=vertical_load*p.section_h*0.5
        #     model.elems[p.cand].ll[0]+=horizontal_load*p.section_h*0.5
        img_scale = 1/p.section_h
        if model.elems[p.cand].type == 'mortar' and model.elems[p.anta].type == 'mortar':
            p.cont_type.type = 'friction_fc_cohesion'
            p.cont_type.fc = data_config['mortar_mortar_fc']*mpa_to_2d  # 10 times ft
            p.cont_type.ft = data_config['mortar_mortar_ft']*mpa_to_2d
            p.cont_type.cohesion = data_config['mortar_mortar_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_to_2d
            p.cont_type.mu = data_config['mortar_mortar_mu']
        elif (model.elems[p.cand].type.startswith('stone') and model.elems[p.anta].type == 'mortar') or\
                (model.elems[p.cand].type == 'mortar' and model.elems[p.anta].type.startswith('stone')):
            if 'different_vertical_horizontal_stone_mortar_interface_fc' not in data_config.keys() or bool(data_config['different_vertical_horizontal_stone_mortar_interface_fc']) == False:
                p.cont_type.type = 'friction_fc_cohesion'
                p.cont_type.fc = data_config['mortar_stone_fc']*mpa_to_2d
                p.cont_type.ft = data_config['mortar_stone_ft']*mpa_to_2d
                p.cont_type.cohesion = data_config['mortar_stone_cohesion'] * \
                    p.section_h*_cohesion_scale*mpa_to_2d
                p.cont_type.mu = data_config['mortar_stone_mu']
            elif bool(data_config['different_vertical_horizontal_stone_mortar_interface_fc']) == True:
                if abs(p.normal[0]) <= abs(p.normal[1]):  # horizontal face
                    p.cont_type.type = 'friction_fc_cohesion'
                    p.cont_type.fc = data_config['mortar_stone_fc']*mpa_to_2d
                    p.cont_type.ft = data_config['mortar_stone_ft']*mpa_to_2d
                    p.cont_type.cohesion = data_config['mortar_stone_cohesion'] * \
                        p.section_h*_cohesion_scale*mpa_to_2d
                    p.cont_type.mu = data_config['mortar_stone_mu']
                else:
                    p.cont_type.type = 'friction_fc_cohesion'
                    p.cont_type.fc = data_config['mortar_stone_fc_v']*mpa_to_2d
                    p.cont_type.ft = data_config['mortar_stone_ft_v']*mpa_to_2d
                    p.cont_type.cohesion = data_config['mortar_stone_cohesion_v'] * \
                        p.section_h*_cohesion_scale*mpa_to_2d
                    p.cont_type.mu = data_config['mortar_stone_mu_v']
        # elif model.elems[p.cand].type == 'ground' or model.elems[p.anta].type == 'ground' or\
        #         model.elems[p.cand].type == 'beam' or model.elems[p.anta].type == 'beam':
        elif model.elems[p.cand].type == 'ground' or model.elems[p.cand].type == 'beam':
            p.cont_type.type = 'friction_fc_cohesion'
            p.cont_type.fc = data_config['ground_beam_fc']*mpa_to_2d
            p.cont_type.ft = data_config['ground_beam_ft']*mpa_to_2d
            p.cont_type.cohesion = data_config['ground_beam_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_to_2d
            p.cont_type.mu = data_config['ground_beam_mu']
        elif model.elems[p.anta].type == 'ground' or model.elems[p.anta].type == 'beam':
            p.cont_type.type = 'friction_fc_cohesion'
            p.cont_type.fc = data_config['ground_beam_fc']*mpa_to_2d  # 10 times ft
            p.cont_type.ft = data_config['ground_beam_fc']*mpa_to_2d
            p.cont_type.cohesion = data_config['ground_beam_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_to_2d
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
            p.cont_type.fc = data_config['stone_stone_fc']*mpa_to_2d
            p.cont_type.ft = data_config['stone_stone_ft']*mpa_to_2d
            p.cont_type.cohesion = data_config['stone_stone_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_to_2d
            p.cont_type.mu = data_config['stone_stone_mu']
        elif model.elems[p.cand].type != model.elems[p.anta].type and model.elems[p.anta].type.startswith('stone') and model.elems[p.cand].type.startswith('stone'):
            p.cont_type.type = 'friction_fc_cohesion'
            p.cont_type.fc = data_config['stone_other_stone_fc']*mpa_to_2d
            p.cont_type.ft = data_config['stone_other_stone_ft']*mpa_to_2d
            p.cont_type.cohesion = data_config['stone_other_stone_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_to_2d
            p.cont_type.mu = data_config['stone_other_stone_mu']
        else:
            print(
                f"Unknow contact between {model.elems[p.cand].type} and {model.elems[p.anta].type}!")
        #apply ground interface to elements closer to the ground
        #otherwise it is very likely that the stone-mortar interface close to the ground is the first to fail
        #which in reality strong mortar is applied at this height
        if p.coor[1] >= ground_y_threshold:
            p.cont_type.fc = data_config['ground_beam_fc']*mpa_to_2d
            p.cont_type.ft = data_config['ground_beam_ft']*mpa_to_2d
            p.cont_type.cohesion = data_config['ground_beam_cohesion'] * \
                p.section_h*_cohesion_scale*mpa_to_2d
            p.cont_type.mu = data_config['ground_beam_mu'] 
    print("number of contact points:", len(model.contps.values()))
    print("number of elments: ", len(model.elems.values()))
    # save model figure
    #write_to_csv(model.elems, model.contps, result_folder)
    # **************************************************************solve associative without crack
    if solver == 'associative':
        elems_1 = copy.deepcopy(model.elems)
        contps_1 = copy.deepcopy(model.contps)
        start = time.time()
        Aglobal = A_matrix
        solution = solve_finitefc_associative(
            elems_1, contps_1, Aglobal=Aglobal)
        end = time.time()

        print('Associative solver time:', end - start)
        time_compute = end - start
        # with open(str(_result_dir)+f'/time.txt', 'a') as f:
        #     f.write(
        #         f'{img_width};{img_height};{nb_element};{nb_contact_point};associative;{vertical_load};{horizontal_load};{end - start}\n')
        # with open(str(_result_dir)+f'/load_multiplier.txt', 'a') as f:
        #     lm = solution['limit_force']
        #     f.write(
        #         f'associative;{vertical_load};{horizontal_load};{lm}\n')
        #print(solution['limit_force'])
        load_multiplier = solution['limit_force']*lm_factor
        for i_e in elems_1.values():
            if abs(i_e.displacement[0]) > max_x_disp:
                max_x_disp = abs(i_e.displacement[0])
        if load_multiplier > 0:
            # identify failure type
            y = solution["Lagrange_multiplier"]
            faceIDs = solution["faceIDs"]
            contfs = solution["contfs"]
            contps_1_keys_loop = []
            for key, value in contps_1.items():
                contps_1_keys_loop.append(key)

            # check y dimension
            nb_constraint = len(elems_1)*3+len(contps_1)*3+len(faceIDs)*8
            if len(y) != nb_constraint:
                print("y dimension error!")

            to_csv_2d(
                elems_1, contps_1, f'vload_{d_vertical_load}_hload_{l_horizontal_load}_{load_multiplier:0.2f}', result_folder)


    return {"load_multiplier": load_multiplier, "time": time_compute, "max_x_disp": max_x_disp}


# *******************************************
# ******** Configuration ********************
# *******************************************
_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)

with open(str(_result_directory)+f'/load_multiplier.txt', 'w+') as f:
    f.write(
        f'image;solver;sampleid;mm_fc;mm_ft;mm_c;mm_mu;mc_fc;mc_ft;mc_c;mc_mu;force_capacity_MPa\n')




solver = "associative"
# ******************************************* rigid body model

def run_Vas_one_test(input):
    [imagename, sampleid, model, A_matrix,max_y] = input
    #print(df[df['sampleid']==sampleid]['mm_fc'])
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

    # set material property
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
    #set ground bound
    ground_y_threshold_ = max_y +data_config['ground_y_threshold_offset']
    # ******************************************* loading
    record = run_one_mesh(model=model, A_matrix=A_matrix, data_folder=kmodel_save_dir, result_folder=current_result_directory,
                        d_vertical_load=vload, l_horizontal_load=1*hload_direction, solver=solver, boundary=data_config['boundary'], data_config=data_config_current,ground_y_threshold = ground_y_threshold_)
    #scale multiplier
    force_capacity_MPa = record["load_multiplier"]/(data_config['scale_img_to_width']*data_config['wall_thickness'])*1000
    print("Maximum force capacity in MPa:", force_capacity_MPa)

    # ******************************************* txt
    # write record to txt file
    with open(str(_result_directory)+f'/load_multiplier.txt', 'a') as f:
        f.write(imagename+';'+solver +
                f';{sampleid};{mm_fc};{mm_ft};{mm_c};{mm_mu};{mc_fc};{mc_ft};{mc_c};{mc_mu};{force_capacity_MPa}\n')

# get model
set_dimension(2)
model = Model()

# ************************************************************** read data
model.from_csv(kmodel_save_dir)

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
min_y = 1e10
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

#calculate A matrix
A_matrix = cal_A_global_2d(model.elems, model.contps, sparse=True)

run_Vas_one_test([imagename, 0, model, A_matrix,max_y])
