# Under developement
# %%
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.patches import Polygon
import json
plt.style.use('science')
chosen_colors = ["#045275","#FCDE9C","#7CCBA2","#F0746E","#7C1D6F"]
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

#read config
config_dir = _data_dir
with open(config_dir+'/config.json') as config_file:
    data_config = json.load(config_file)

scale_x = data_config['scale_to_m_x']
scale_y = data_config['scale_to_m_y']


result_dir = current_result_dir
folder = imagename.split(".")[0]
directory = 'run_limit_analysis/'+folder+'_sample0'

def split_x_y_value(df):
    df[['x', 'y']] = df['coor'].str.split(",", expand=True)
    df['xcoord'] = df['x'].str.split("[", expand=True)[1].astype(float)
    df['ycoord'] = df['y'].str.split("]", expand=True)[0].astype(float)
    df[['disp_x', 'disp_y']] = df['displacement'].str.split(",", expand=True)
    df['disp_x'] = df['disp_x'].str.split("[", expand=True)[1].astype(float)
    df['disp_y'] = df['disp_y'].str.split("]", expand=True)[0].astype(float)
    return df

def split_x_y_value_element(df):
    df[['x', 'y']] = df['center'].str.split(",", expand=True)
    df['xcoord'] = df['x'].str.split("[", expand=True)[1].astype(float)
    df['ycoord'] = df['y'].str.split("]", expand=True)[0].astype(float)
    df[['disp_x', 'disp_y','disp_r']] = df['displacement'].str.split(",", expand=True)
    df['disp_x'] = df['disp_x'].str.split("[", expand=True)[1].astype(float)
    df['disp_y'] = df['disp_y'].astype(float)
    df['disp_r'] = df['disp_r'].str.split("]", expand=True)[0].astype(float)
    return df
import json
def get_shape_vertices_from_file(df,jaso_file_dir):
    """
    read the shape file from the df for each line of element
    return th dictionary that map elemetn id with shape vertices [[x,y],[x,y]...]
    """
    shape_dict = {}
    for index, row_element in df.iterrows():
        shape_dict[row_element['id']] = []
        #read json file
        with open(jaso_file_dir+'/'+row_element['shape_file']) as f:
            shape_dict[row_element['id']] = json.load(f)
    return shape_dict
import math
from matplotlib import collections as mc

def plot_crack_map(df,df_element, save_file_name, wall_image_size,disp_factor = 1,shape_dict = None):
    plt.clf()
    df = df[['id','xcoord','ycoord','disp_x','disp_y','cand','counterPoint','faceID']]
    df_element = df_element[['id','type','xcoord','ycoord','disp_x','disp_y','disp_r','shape_file']]

    fig, ax = plt.subplots()
    #plot elements
    max_element_plot = 1e10
    lines = []
    lines_stones = []
    lines_bound = []
    xs = []
    ys = []
    if not shape_dict is None:
        for index, row_element in df_element.iterrows():
            if index>max_element_plot:
                break
            boundary_points = []
            polygon_pts = []
            center = [row_element['xcoord'],row_element['ycoord']]
            trans_x = row_element['disp_x']*disp_factor
            trans_y = row_element['disp_y']*disp_factor
            rot = row_element['disp_r']*disp_factor

            vertices  =shape_dict[row_element['id']]
            # if row_element['type'] == 'beam':
            #     continue
            #     # add the left on the top point to the begining of the list
            #     vertices.insert(0,[vertices[0][0],-0.001])
            #     # add the right on the top point to the end of the list
            #     vertices.append([vertices[-1][0],-0.001])
            # elif row_element['type'] == 'ground':
            #     # add the left on the bottom point to the begining of the list
            #     vertices.insert(0,[vertices[0][0],0.7])
            #     # add the right on the bottom point to the end of the list
            #     vertices.append([vertices[-1][0],0.7])
            
            shade_x = []
            shade_y = []

            for pt in vertices:
                node_x = pt[0]*scale_x-center[0]
                node_y = pt[1]*scale_y-center[1]

                new_x = node_x*math.cos(rot)+node_y * \
                    math.sin(rot)+trans_x+center[0]
                new_y = -node_x*math.sin(rot)+node_y * \
                    math.cos(rot)+trans_y+center[1]
                boundary_points.append((new_x, new_y))
                polygon_pts.append([new_x,new_y])
                # # shade mortar
                # shade_x.append(new_x)
                # shade_y.append(new_y)
                # if row_element['type'].startswith('stone'):
                #         xs.append(new_x)
                #         ys.append(new_y)
            for i in range(len(boundary_points)):
                lines.append([boundary_points[i-1], boundary_points[i]])
            polygon_pts = np.asarray(polygon_pts)
            if row_element['type'].startswith('stone'):
                polycolor = 'black'
            else:
                polycolor = 'grey'
            poly = Polygon(polygon_pts, closed=True, edgecolor='none', facecolor=polycolor)
            ax.add_patch(poly)

    merged_df = pd.merge(df, df_element, left_on='cand',
                         right_on='id', suffixes=('_cp', '_el'))
    
    merged_df = merged_df[['id_cp','xcoord_cp','ycoord_cp','disp_x_cp','disp_y_cp','cand','counterPoint','faceID','type','xcoord_el','ycoord_el','disp_x_el','disp_y_el','disp_r']]
    merged_df = pd.merge(merged_df, merged_df, left_on='counterPoint',
                         right_on='id_cp', suffixes=('_master', '_slave'))
    merged_df['displacement_difference']= np.sqrt((merged_df['disp_x_cp_master']-merged_df['disp_x_cp_slave'])**2+(merged_df['disp_y_cp_master']-merged_df['disp_y_cp_slave'])**2)
    max_point_crack_width = merged_df['displacement_difference'].max()


    seismic = plt.get_cmap('hot', 200)
    

    

    for face_id in range(0,df['faceID'].max()):
        if face_id>max_element_plot:
            break
        # calculate displaced x y of contact points
        current_face_df=  merged_df[merged_df['faceID_master']==face_id]
        center_xs = current_face_df['xcoord_el_master']
        if center_xs.empty:
            continue
        center_ys = current_face_df['ycoord_el_master']
        node_xs = current_face_df['xcoord_cp_master']-center_xs
        node_ys = current_face_df['ycoord_cp_master']-center_ys
        rots = current_face_df['disp_r_master']*disp_factor
        trans_xs = current_face_df['disp_x_el_master']*disp_factor
        trans_ys = current_face_df['disp_y_el_master']*disp_factor
        new_xs = node_xs*rots.apply(np.cos)+node_ys * \
                    rots.apply(np.sin)+trans_xs+center_xs
        new_ys = -node_xs*rots.apply(np.sin)+node_ys * \
            rots.apply(np.cos)+trans_ys+center_ys

        #plot
        start_point = [new_xs.iloc[[0]],new_ys.iloc[[0]]]
        end_point = [new_xs.iloc[[1]],new_ys.iloc[[1]]]
        averaged_face_disp = current_face_df['displacement_difference'].mean()
        # if current_face_df['type_master'].str.contains('beam').any() or current_face_df['type_slave'].str.contains('beam').any():
        #     averaged_face_disp = 0
        # if current_face_df['type_master'].str.contains('beam').any():
        #     continue
        # ax.plot([start_point[0], end_point[0]],
        #         [start_point[1], end_point[1]], color=seismic(
        #     round(100 * (averaged_face_disp/max_point_crack_width))), lw=0.2+0.5*(averaged_face_disp/max_point_crack_width))
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]], color='red', \
                    lw=0.5,\
                        alpha = 1*(averaged_face_disp/max_point_crack_width))
    
    min_x = 0
    min_y = 0
    max_x = wall_image_size[1]

    max_y = wall_image_size[0]
    xscale = max_x-min_x
    yscale = max_y-min_y
    plt.xlim(min_x-0.1*xscale, max_x+0.1*xscale)
    plt.ylim(min_y-0.1*yscale, max_y+0.1*yscale)

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    

    plt.savefig(save_file_name+'_displacement.png', dpi = 600,transparent=True)
    plt.close()


save_file_dir = result_dir+'/'+folder+'/'+directory+'/'
if not os.path.exists(save_file_dir):
    os.system('mkdir '+save_file_dir)
for point_file in glob.iglob(result_dir+'/'+folder+'/'+directory+'/vload_*_hload_*_contact_point.csv',recursive=True):
    
    point_data = pd.read_csv(point_file)
    element_data = pd.read_csv(glob.glob(result_dir+'/'+folder+'/'+directory+'/vload_*_hload_*_element.csv')[0])
    save_file_name = save_file_dir

    point_data = split_x_y_value(point_data)
    element_data = split_x_y_value_element(element_data)
    shape_file_dir = result_dir+'/'+folder+'/'+'kmodel'+'/'
    shape_dict = get_shape_vertices_from_file(element_data, shape_file_dir)
    max_x = data_config['scale_img_to_width']
    max_y = data_config['scale_img_to_height']
    # get disp_factor
    # the maximum displacement is 1/wall_thickness
    disp_factor = data_config['plot_max_disp']*data_config['wall_thickness']* data_config['scale_to_m_x']
    plot_crack_map(point_data,element_data, save_file_name, (max_x,max_y),disp_factor = disp_factor,shape_dict = shape_dict)

