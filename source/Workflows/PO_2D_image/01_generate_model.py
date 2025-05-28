"""
Description: This file contains the functions to convert the binary images into rigid block models
Input: binary image ("SW1_0074_REF.png") and configuration file (config.json)
        The images and config file are stored in the folder "data_04"
Output: rigid block model stored in the folder "result_04"
"""
import cv2
from datetime import datetime
import gmsh
import json
from Kinematics import *
import math
import matplotlib.pyplot as plt
import meshio
import numpy as np
import os
import pathlib
from rtree import index
from scipy.spatial import ConvexHull
from shapely import affinity
from shapely.geometry import Polygon, Point
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import regionprops, find_contours
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error
import skimage
from threadpoolctl import ThreadpoolController
import time
import tqdm
import open3d as o3d

plt.style.use('science')
controller = ThreadpoolController()
DEBUG = False
global GROUND_ID_IN_MATRIX
global BEAM_ID_IN_MATRIX
global GROUND_ID_IN_MODEL
global BEAM_ID_IN_MODEL
EPS = 1e-9
BETA = 0

_root_dir = pathlib.Path(__file__).resolve().parent
_root_dir = os.path.abspath(_root_dir)
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

def cal_anta_id_with_rtree(points, tree_tuple):
    proper_tree = index.Property(dimension=4)
    idx_normal_cand_points = index.Index(
        interleaved=False, properties=proper_tree)
    new_points = dict()
    for p_cand in tqdm.tqdm(points.values()):
        if p_cand.counterPoint is not None:
            continue
        near_point_ids = list(tree_tuple[1].nearest((p_cand.coor[0], p_cand.coor[0], p_cand.coor[1], p_cand.coor[1],
                              p_cand.normal[0], p_cand.normal[0], p_cand.normal[1], p_cand.normal[1], p_cand.section_h, p_cand.section_h), 1))

        if len(near_point_ids) == 0:
            continue
        # check if all the points found are valid anta points, not duplicate
        cand_ids_for_check = []
        for i_e in range(len(near_point_ids)):
            if near_point_ids[i_e] in points.keys():
                cand_ids_for_check.append(points[near_point_ids[i_e]].cand)
        if len(set(cand_ids_for_check)) != len(cand_ids_for_check):
            print(
                "More than one counter point is found on the same cand element")
        # print(
        #     "More than one counter point is found, they belong to different cand element")
        for i_c in range(len(near_point_ids)):
            if not near_point_ids[i_c] in points.keys():
                continue

            p_anta = points[near_point_ids[i_c]]
            if p_anta.counterPoint is not None:
                continue
            if not p_anta.is_contat_pair(p_cand):
                continue
            else:
                p_cand.anta = p_anta.cand
                p_cand.counterPoint = p_anta.id
                p_anta.counterPoint = p_cand.id
                p_anta.anta = p_cand.cand
                new_points[p_cand.id] = p_cand
                new_points[p_anta.id] = p_anta
                idx_normal_cand_points.insert(
                    p_cand.id, (p_cand.normal[0], p_cand.normal[0], p_cand.normal[1], p_cand.normal[1], p_cand.cand, p_cand.cand, p_cand.anta, p_cand.anta))
                idx_normal_cand_points.insert(
                    p_anta.id, (p_anta.normal[0], p_anta.normal[0], p_anta.normal[1], p_anta.normal[1], p_anta.cand, p_anta.cand, p_anta.anta, p_anta.anta))
                break

    return new_points, idx_normal_cand_points

def cal_center(points):
    center_x = 0
    center_y = 0
    for p in points:
        center_x += p[0]
        center_y += p[1]
    return [center_x/len(points), center_y/len(points)]


def cal_tri_area(three_points):
    a = np.sqrt((three_points[0][0]-three_points[1][0])**2 +
                (three_points[0][1]-three_points[1][1])**2)
    b = np.sqrt((three_points[1][0]-three_points[2][0])**2 +
                (three_points[1][1]-three_points[2][1])**2)
    c = np.sqrt((three_points[2][0]-three_points[0][0])**2 +
                (three_points[2][1]-three_points[0][1])**2)
    s = (a+b+c)/2
    return np.sqrt(s*(s-a)*(s-b)*(s-c))

def nn_sort(points):
    """Sort order of points based on nn search. Could fail if the shape is flat.
    """
    unvisited_points = points.copy()
    sorted_points = []
    sorted_points.append(unvisited_points[0])
    unvisited_points = np.delete(unvisited_points, 0, axis=0)
    while len(unvisited_points) > 0:
        last_point = sorted_points[-1]
        distances = np.linalg.norm(
            unvisited_points - last_point, axis=1)
        nearest_point_index = np.argmin(distances)
        nearest_point = unvisited_points[nearest_point_index]
        sorted_points.append(nearest_point)
        unvisited_points = np.delete(unvisited_points, nearest_point_index, axis=0)
    return np.array(sorted_points)

def clockwise_sort(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # check if points are linear
    if get_linear_regression_score(points) > 0.9:
        centroid = np.mean(points, axis=0)+np.asarray([0, 1])

    # Calculate the angles of each point relative to the centroid
    angles = np.arctan2(points[:, 1]-centroid[1], points[:, 0]-centroid[0])

    # Check if the angles are unique
    if len(np.unique(angles)) != len(angles):
        # delete the point with the same angle
        unique_angles = np.unique(angles)
        unique_points = []
        for angle in unique_angles:
            unique_points.append(points[np.where(angles == angle)[0][0]])
        points = np.array(unique_points)
        angles = unique_angles

    # Sort the points by their angles
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    # Reverse the order of the points to get them in clockwise order
    clockwise_points = sorted_points[::-1]

    # calculate the perimeter of the polygon
    perimeter = 0
    for i in range(len(clockwise_points)-1):
        perimeter += np.linalg.norm(
            clockwise_points[i+1]-clockwise_points[i])
    perimeter += np.linalg.norm(
        clockwise_points[0]-clockwise_points[-1])
    
    # compare with nnsort
    nn_sorted_points = nn_sort(points)
    ## calculate the perimeter of the nn_sorted_points
    nn_sorted_perimeter = 0
    for i in range(len(nn_sorted_points)-1):
        nn_sorted_perimeter += np.linalg.norm(
            nn_sorted_points[i+1]-nn_sorted_points[i])
    nn_sorted_perimeter += np.linalg.norm(
        nn_sorted_points[0]-nn_sorted_points[-1])
    if nn_sorted_perimeter < perimeter:
        return nn_sorted_points


    return clockwise_points



def create_elem_contp_from_vertices(elems, contps, idx_contps_reverse, contp_index, face_id, element_id, center, mass, vertices, type, _conttype):
    # It's important to shift the coordinate of vertices so that no confusion in contact pair detection in case of multiple elements sqreezing on one point
    shape_file = f"element_{element_id}.json"
    elems[element_id] = Element(
        element_id, center, mass, vertices, type=type, shape_file=shape_file)
    
    return elems, contps, idx_contps_reverse, contp_index, face_id

@controller.wrap(limits=1, user_api='blas')
def get_linear_regression_score(vertices):
    # regression score with origional points
    reg_ori = LinearRegression().fit(
        vertices[:, 0].reshape(-1, 1), vertices[:, 1])
    pred_ori = reg_ori.predict(vertices[:, 0].reshape(-1, 1))
    #score_ori = r2_score(vertices[:, 1], pred_ori)
    score_ori = reg_ori.score(
        vertices[:, 0].reshape(-1, 1), vertices[:, 1])
    # regression score with xy swapped points
    reg_swap = LinearRegression().fit(
        vertices[:, 1].reshape(-1, 1), vertices[:, 0])
    score_swap = reg_swap.score(
        vertices[:, 1].reshape(-1, 1), vertices[:, 0])
    pred_swap = reg_swap.predict(vertices[:, 1].reshape(-1, 1))
    R2 = max(score_ori, score_swap)
    MSE = min(mean_squared_error(vertices[:, 1], pred_ori), mean_squared_error(
        vertices[:, 0], pred_swap))
    if MSE < 0.0001:
        R2 = 1
    return R2


def move_one_point_to_another(p1, p2, shift=0.01):
    # move p1 to p2
    # p1 and p2 are np.array([x,y])
    # shift is the distance to move
    # return the new position of p1
    if p1.tolist() == p2.tolist():
        raise ValueError("Two points on the same contact face overlapps.")
    else:
        return p1 + (p2-p1)*shift/np.linalg.norm(p2-p1)

def downsample_voxel_onesize(vertices, mesh_size):

    print(
        f"Downsample the point cloud ({vertices.shape[0]}) with a voxel of {mesh_size}")

    # add 3rd dimension to vertices array
    vertices = np.hstack(
        (vertices, np.zeros((vertices.shape[0], 1))))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    downpcd = pcd.voxel_down_sample(voxel_size=mesh_size)
    vertices_down = np.asarray(downpcd.points)
    # remove the 3rd dimension
    vertices_down = vertices_down[:, 0:2]
    print(
        f"After downsampling, the point cloud has {vertices_down.shape[0]} points")

    return vertices_down

def pixel_to_elem_polystone(matrix, scale=1, density=1, contact_type_dict={"type_name": "friction"}, contp_start_index=0, face_start_index=0, downsample_method='fft', point_grid_size=10, number_segments=10, _result_directory=None,plot_edge_detection = True):
    # element id equals pixel value
    proper_tree = index.Property(dimension=2)
    idx_polys = index.Index(interleaved=False, properties=proper_tree)
    proper_tree = index.Property(dimension=5)
    idx_contps_reverse = index.Index(interleaved=False, properties=proper_tree)
    matrix_id = np.zeros(matrix.shape).astype(np.int16)
    matrix_type = np.zeros(matrix.shape)
    elems = dict()
    max_element_id = -np.inf
    contps = dict()
    max_contp_id = -np.inf
    _conttype = ContType(
        contact_type_dict['type_name'], contact_type_dict)
    face_id = face_start_index
    labels = skimage.measure.label(matrix, connectivity=2)
    contp_index = contp_start_index
    if plot_edge_detection:
        tab20 = plt.get_cmap('tab20', 20)
        fig, axs = plt.subplots(1, 2)

    max_deviation = 0
    max_deviation_position = []
    min_x = np.inf
    max_x = -np.inf
    for region in regionprops(labels):
        # if region.label==35:
        #     plt.imshow(region.image)
        #     plt.show()
        center = [region.centroid[1]*scale, region.centroid[0]*scale]
        if matrix[int(region.centroid[0]*scale), int(region.centroid[1]*scale)] == BEAM_ID_IN_MATRIX:
            _type = "beam"
            global BEAM_ID_IN_MODEL
            BEAM_ID_IN_MODEL = region.label
        elif matrix[int(region.centroid[0]*scale), int(region.centroid[1]*scale)] == GROUND_ID_IN_MATRIX:
            _type = "ground"
            global GROUND_ID_IN_MODEL
            GROUND_ID_IN_MODEL = region.label
        else:
            _type = f'stone_{region.label}'
        element_id = region.label
        # print(element_id)
        if element_id in list(elems.keys()):
            print("Duplicate element id:", element_id)
        area = region.area*(scale**2)
        mass = area

        region_mask = np.zeros(matrix.shape)
        region_mask[region.bbox[0]:region.bbox[2],
                    region.bbox[1]:region.bbox[3]] = region.image_filled
        # plt.imshow(region_mask)
        # plt.show()

        contour = find_contours(region_mask, level=0.5)[0]
        contour_flipped = np.flip(contour, axis=1)*scale
        vertices = contour_flipped

        # delete repeated vertices
        if vertices[0].tolist() == vertices[-1].tolist():
            # delete the last row
            vertices = vertices[0:-1]
        if plot_edge_detection:
            color = tab20(element_id % 20)
            axs[0].plot(vertices[:, 0], vertices[:, 1], 'o-',
                        linewidth=0.2, markersize=0.1, color=color)
            query_points = vertices.copy()
        # down sample the vertices
        ###############################
        # downsample using voxel-grid #
        ###############################

        def dowmsample_interval_onesize(vertices, interval=10):
            vertices_down = vertices[::interval, :]
            return vertices_down

        def downsample_interval(vertices, interval=10):
            # downsample using uniform interval
            query_points = vertices.copy()
            ########################################################################
            # check colinearity of downsampled points
            ########################################################################
            # check if the points are on the same line
            score_ori = get_linear_regression_score(vertices)
            vertices_down = dowmsample_interval_onesize(
                vertices, interval=interval)
            # check if the points are on the same line
            score_down = get_linear_regression_score(vertices_down)
            _line_threshold = 0.9
            while (score_down > _line_threshold and score_ori <= 0.9) or (score_ori <= 0.9 and vertices_down.shape[0] <= 2):
                interval = math.ceil(interval/2)
                print(
                    f"The points are on the same line after downsample, downsample with a smaller interval size {interval}")
                if interval <= 1:
                    vertices_down = vertices
                    break
                vertices_down = dowmsample_interval_onesize(vertices, interval)
                # check if the points are on the same line
                score_down = get_linear_regression_score(vertices_down)
                #vertices = vertices[:, 0:2]
            else:
                # check if the element is ground/beam, if so, add the start and end points
                if _type == "beam" or _type == "ground":
                    vertices = vertices[np.argsort(
                        np.linalg.norm(vertices, axis=1))]
                    vertices_down = vertices_down[np.argsort(
                        np.linalg.norm(vertices_down, axis=1))]
                    if vertices[0, 0:2].tolist() != vertices_down[0].tolist():
                        # add the first point
                        vertices_down = np.vstack(
                            (vertices[0, 0:2], vertices_down))
                    if vertices[0, 0] > 0:
                        vertices_down = np.vstack(
                            (np.asarray[0, vertices[0, 1]], vertices_down))
                    if vertices[-1, 0:2].tolist() != vertices_down[-1].tolist():
                        # add the last point
                        vertices_down = np.vstack(
                            (vertices_down, vertices[-1, 0:2]))
                    if vertices[-1, 0] < matrix.shape[1]*scale:
                        vertices_down = np.vstack(
                            (vertices_down, np.asarray([matrix.shape[1]*scale, vertices[-1, 1]])))
                    vertices_down = vertices_down[np.argsort(
                        np.linalg.norm(vertices_down, axis=1))]
                    # order the vertices
                    vertices_down = clockwise_sort(vertices_down)
                else:  # add the corner points
                    # add corner points
                    corner_coors = corner_peaks(
                        corner_harris(region_mask), min_distance=interval, threshold_rel=0.02)
                    corner_coors = np.flip(corner_coors, axis=1)*scale
                    for coor in corner_coors:
                        # check if the corner point is in the new_vertices
                        if coor.tolist() not in vertices_down.tolist():
                            vertices_down = np.vstack((vertices_down, coor))
                    # order the vertices
                    vertices_down = clockwise_sort(vertices_down)
                    
                    
            vertices = vertices_down
            max_deviation = 0
            max_deviation_position = []

            if vertices.shape[0] > 2:
                query_poly = Polygon(vertices)
                for p in query_points:
                    if query_poly.distance(Point(p[0], p[1])) > max_deviation:
                        max_deviation = query_poly.distance(
                            Point(p[0], p[1]))
                        max_deviation_position = p
            else:
                max_deviation = 0
            return vertices, (max_deviation, max_deviation_position)

        def downsample_voxel(vertices, _type=_type, convex_hull=True):
            query_points = vertices.copy()

            # check if the points are on the same line
            score_ori = get_linear_regression_score(vertices)
            mesh_size = point_grid_size
            vertices_down = downsample_voxel_onesize(vertices, mesh_size)
            # check if the points are on the same line
            score_down = get_linear_regression_score(vertices_down)
            _line_threshold = 0.9
            # case where score is 0.53 as a line:
            # array([[ 6.95      ,  9.5       ],
            #         [19.        ,  9.5       ],
            #         [43.66666667,  9.41666667],
            #         [34.        ,  9.5       ]])

            while (score_down > _line_threshold and score_ori <= 0.9) or (score_ori <= 0.9 and vertices_down.shape[0] <= 2):
                print(
                    "The points are on the same line after downsample, downsample with a smaller voxel size")
                mesh_size = math.ceil(mesh_size/2)
                if mesh_size <= 1:
                    vertices_down = vertices
                    break
                vertices_down = downsample_voxel_onesize(vertices, mesh_size)
                # check if the points are on the same line
                score_down = get_linear_regression_score(vertices_down)
                #vertices = vertices[:, 0:2]
            else:
                # check if the element is ground/beam, if so, add the start and end points
                if _type == "beam" or _type == "ground":
                    vertices = vertices[np.argsort(
                        np.linalg.norm(vertices, axis=1))]
                    vertices_down = vertices_down[np.argsort(
                        np.linalg.norm(vertices_down, axis=1))]
                    if vertices[0, 0:2].tolist() != vertices_down[0].tolist():
                        # add the bound points
                        vertices_down = np.vstack(
                            (vertices[0, 0:2], vertices_down))
                    if vertices[-1, 0:2].tolist() != vertices_down[-1].tolist():
                        # add the bound points
                        vertices_down = np.vstack(
                            (vertices_down, vertices[-1, 0:2]))
                else:  # add the four corner points
                    # add corner points
                    corner_coors = corner_peaks(
                        corner_harris(region_mask), min_distance=1, threshold_rel=0.02)
                    corner_coors = np.flip(corner_coors, axis=1)*scale
                    for coor in corner_coors:
                        # check if the corner point is in the new_vertices
                        if coor.tolist() not in vertices_down.tolist():
                            vertices_down = np.vstack((vertices_down, coor))
                vertices = vertices_down
            # order the vertices
            try:
                if convex_hull:
                    hull = ConvexHull(vertices)
                    # in counterclockwise order
                    vertices = vertices[hull.vertices]
                else:
                    vertices = clockwise_sort(vertices)
            except:
                print(
                    "Convex hull/Angle sort failed, order vertices by distance to origin")
                vertices = vertices[np.argsort(
                    np.linalg.norm(vertices, axis=1))]
            max_deviation = 0
            query_poly = Polygon(vertices)
            for p in query_points:
                if query_poly.distance(Point(p[0], p[1])) > max_deviation:
                    max_deviation = query_poly.distance(
                        Point(p[0], p[1]))

            return vertices, max_deviation

        ########################
        # downsample using FFT #
        ########################
        from scipy import interpolate

        def resample_uniform_signals(signal, points_num=2**8):
            _signal_x = np.linspace(1, len(signal), len(signal))
            new_signal_x = np.linspace(
                min(_signal_x), max(_signal_x), points_num)
            fcubic = interpolate.interp1d(
                _signal_x, signal, kind='cubic')  # , kind='cubic'
            new_signal = fcubic(new_signal_x)
            return new_signal_x, new_signal

        def expand_signal_1D(signal_1D):
            coefs = np.fft.fft(signal_1D, norm=None)
            return coefs

        def reconstruct_1D_signal(coefs, max_freq):
            window = np.zeros(len(coefs))
            window[:max_freq] = 1
            rec_signal = np.fft.ifft(coefs*window, norm=None)
            return rec_signal.real

        def downsample_fft(vertices, _type=_type, segments=10, max_rec2=7):
            polygon_contour = Polygon(vertices).length
            if not _type == "beam" or _type == "ground":
                segments = max(segments, int(polygon_contour/point_grid_size))
            if segments != 10:
                print("debug")
                pass
            query_points = vertices.copy()
            # remove the mean
            ori_mean = np.mean(vertices, axis=0)
            new_vertices = vertices - ori_mean
            # remove scale for both dimensions
            ori_scale = np.max(
                new_vertices, axis=0)-np.min(new_vertices, axis=0)
            if ori_scale[0] != 0:
                new_vertices[:, 0] = new_vertices[:, 0]/ori_scale[0]
            if ori_scale[1] != 0:
                new_vertices[:, 1] = new_vertices[:, 1]/ori_scale[1]
            # close the loop
            new_vertices = np.vstack((vertices, vertices[0]))
            # uniform resample
            num_points = 2**10
            _, res_x = resample_uniform_signals(new_vertices[:, 0], num_points)
            _, res_y = resample_uniform_signals(new_vertices[:, 1], num_points)
            new_vertices = np.zeros([num_points, 2])
            new_vertices[:, 0] = res_x
            new_vertices[:, 1] = res_y
            # downsample

            coefs_x = expand_signal_1D(new_vertices[:, 0])
            coefs_y = expand_signal_1D(new_vertices[:, 1])

            coefs_x = coefs_x[0:segments]
            coefs_y = coefs_y[0:segments]

            rec_x2 = reconstruct_1D_signal(coefs_x, max_rec2)
            rec_y2 = reconstruct_1D_signal(coefs_y, max_rec2)

            # Remove the mean (place the centroid at [0,0])
            rec_x2 -= np.mean(rec_x2)
            rec_y2 -= np.mean(rec_y2)

            # Normalize the output signal
            scale_x2 = max(rec_x2) - min(rec_x2)
            scale_y2 = max(rec_y2) - min(rec_y2)
            if scale_x2 != 0:
                rec_x2 /= scale_x2
            if scale_y2 != 0:
                rec_y2 /= scale_y2

            # recover the mean
            new_vertices = np.zeros([len(coefs_x), 2])
            new_vertices[:, 0] = rec_x2*ori_scale[0]+ori_mean[0]
            new_vertices[:, 1] = rec_y2*ori_scale[1]+ori_mean[1]

            #new_vertices = new_vertices[new_vertices[:, 0] >= 0]
            # check if the element is ground/beam, if so, add the start and end points
            if _type == "beam" or _type == "ground":

                vertices = vertices[np.argsort(
                    np.linalg.norm(vertices, axis=1))]
                new_vertices = new_vertices[np.argsort(
                    np.linalg.norm(new_vertices, axis=1))]
                # remove the first and the last point
                new_vertices = new_vertices[1:-1]
                if vertices[0, 0:2].tolist() != new_vertices[0].tolist():
                    # add the bound points
                    new_vertices = np.vstack(
                        (vertices[0, 0:2], new_vertices))
                if vertices[-1, 0:2].tolist() != new_vertices[-1].tolist():
                    # add the bound points
                    new_vertices = np.vstack(
                        (new_vertices, vertices[-1, 0:2]))
            else:  # add the four corner points
                # add corner points
                corner_coors = corner_peaks(
                    corner_harris(region_mask), min_distance=1, threshold_rel=0.02)
                corner_coors = np.flip(corner_coors, axis=1)*scale
                for coor in corner_coors:
                    # check if the corner point is in the new_vertices
                    if coor.tolist() not in new_vertices.tolist():
                        new_vertices = np.vstack((new_vertices, coor))
            #     # find four points in vertices that maximize the area of the evelopping polygon using Monte Carlo
            #     i_corner = 0
            #     max_area = 0
            #     corner_vertices = new_vertices[0:4]
            #     while i_corner < 1000:
            #         i_corner += 1
            #         # sample four points
            #         sample_points = np.random.choice(
            #             len(vertices), 4, replace=False)
            #         sample_points = vertices[sample_points]
            #         # check if the polygon is convex
            #         if Polygon(sample_points).is_valid:
            #             # check if the polygon is the largest
            #             if Polygon(sample_points).area > max_area:
            #                 max_area = Polygon(sample_points).area
            #                 corner_vertices = sample_points
            #     # add the four corner points if they are not in the new_vertices
            #     for i in range(4):
            #         if corner_vertices[i].tolist() not in new_vertices.tolist():
            #             new_vertices = np.vstack(
            #                 (new_vertices, corner_vertices[i]))

            vertices = new_vertices
            # order the vertices
            try:
                hull = ConvexHull(vertices)
                vertices = vertices[hull.vertices]  # in counterclockwise order
            except:
                print("Convex hull failed, order vertices by distance to origin")
                vertices = vertices[np.argsort(
                    np.linalg.norm(vertices, axis=1))]
            # caluclate deviation
            query_poly = Polygon(vertices)
            max_deviation = 0
            for p in query_points:
                if query_poly.distance(Point(p[0], p[1])) > max_deviation:
                    max_deviation = query_poly.distance(
                        Point(p[0], p[1]))
            return vertices, max_deviation

        if downsample_method == "fft":
            vertices, deviation = downsample_fft(
                vertices, segments=number_segments)
        elif downsample_method == "voxel":
            vertices, deviation = downsample_voxel(vertices, _type=_type)
        elif downsample_method == "interval":
            vertices, deviation = downsample_interval(
                vertices, interval=point_grid_size)
        # if deviation[0] > max_deviation:
        #     max_deviation = deviation[0]
        #     max_deviation_position = deviation[1]

        # For meshing, move all stone vertices towards the center by 0.1
        if _type == "beam" or _type == "ground":
            vertices_min = np.min(vertices, axis=0)
            vertices_max = np.max(vertices, axis=0)
            #vertices = vertices.tolist()
        else:
            ########################################################################
            # check equality of block area of downsampled points
            ########################################################################
            polygon_down = Polygon(vertices)
            area_down = polygon_down.area
            #length = polygon_down.length
            #expand_thickness = (area-area_down)/length
            expand_ratio = np.sqrt(area/area_down)
            #print(expand_ratio)
            polygon_scale = affinity.scale(polygon_down, xfact=expand_ratio,
                                           yfact=expand_ratio, origin='centroid')
            vertices = np.array(polygon_scale.exterior.coords)
            # bound the vertices
            #print(vertices[vertices[:, 0] < EPS][:, 0])
            # a = input("debug")
            # print(vertices)
            vertices[:, 0][vertices[:, 0] < EPS] = EPS
            # print(vertices)
            vertices[:, 1][vertices[:, 1] < EPS] = EPS
            vertices[:, 0][vertices[:, 0] > matrix.shape[1] *
                           scale-EPS] = matrix.shape[1]*scale - EPS
            vertices[:, 1][vertices[:, 1] > matrix.shape[0] *
                           scale-EPS] = matrix.shape[0]*scale - EPS

            # reorder the vertices
            vertices = clockwise_sort(vertices)

            # center = np.array(center)
            # iteration_counter = 0
            # while(expand_thickness > 0 and iteration_counter < 100):
            #     # expand the vertices
            #     polygon_scale = affinity.scale(polygon_down, xfact=expand_ratio,
            #                    yfact=expand_ratio, orgin='center')
            #     vertices = np.array(polygon_scale.exterior.coords)
            #     vertices = vertices + (vertices-center)*expand_thickness / \
            #         np.linalg.norm(vertices-center)
            #     # reorder the vertices
            #     vertices = clockwise_sort(vertices)
            #     # recalculate the area
            #     polygon_down = Polygon(vertices)
            #     area_down = polygon_down.area
            #     length = polygon_down.length
            #     expand_thickness = (area-area_down)/length
            #     iteration_counter += 1

            vertices_min = np.min(vertices, axis=0)
            vertices_max = np.max(vertices, axis=0)
        if plot_edge_detection:
            # add the first point to the end to close the polygon
            plot_vertices = np.vstack((vertices, vertices[0]))
            axs[1].plot(plot_vertices[:, 0], plot_vertices[:, 1], 'o-',
                        linewidth=0.2, markersize=0.1, color=color)

            if not (_type == "beam" or _type == "ground"):
                if vertices.shape[0] > 2:
                    query_poly = Polygon(vertices)
                    for p in query_points:
                        if query_poly.distance(Point(p[0], p[1])) > max_deviation:
                            max_deviation = query_poly.distance(
                                Point(p[0], p[1]))
                            max_deviation_position = p
        

        vertices = vertices.tolist()
        if _type == "beam":
            matrix_type = np.where(matrix_id == element_id, 3, matrix_type)
        elif _type == "ground":
            matrix_type = np.where(matrix_id == element_id, 2, matrix_type)
        else:
            matrix_type = np.where(region_mask != 0, 0, matrix_type)

        elems, contps, idx_contps_reverse, contp_index, face_id = create_elem_contp_from_vertices(
            elems, contps, idx_contps_reverse, contp_index, face_id, element_id, center, mass, vertices, _type, _conttype)
        max_element_id = max(max_element_id, element_id)
        max_contp_id = max(max_contp_id, contp_index)
        matrix_id = np.where(region_mask != 0, element_id, matrix_id)

        if center[0] < min_x:
            min_x = center[0]
        if center[0] > max_x:
            max_x = center[0]
    if plot_edge_detection:
        # plot image
        matrix_bi = 255*(matrix != 0)
        
        axs[0].imshow(matrix_bi, cmap='gray', alpha=0.25,
                      interpolation='none', resample=False)
        axs[1].imshow(matrix_bi, cmap='gray', alpha=0.25,
                      interpolation='none', resample=False)
        # invert y axis for all subplots
        for ax in axs:
            ax.invert_yaxis()
            ax.set_aspect(1)
            # ax.set_axis_off()
            ax.set_xlim(0, matrix.shape[1])
            ax.set_ylim(0, matrix.shape[0])
            #ax.set_xlim(-0.5, matrix.shape[1]-0.5)
            #ax.set_ylim(-0.5, matrix.shape[0]-0.5)

        # add text on top of the figure
        axs[0].text(1, 1.1, 'Deviation: {:.2f} mm, Position {:.2f}, {:.2f}'.format(max_deviation, max_deviation_position[0], max_deviation_position[1]), fontsize=8,
                    horizontalalignment='center', transform=axs[0].transAxes)
        # remove axis
        axs[0].axis('off')
        axs[1].axis('off')
        # reverse y axis
        axs[0].invert_yaxis()
        axs[1].invert_yaxis()

        plt.savefig(_result_directory+"/edge_detection.png",
                    dpi=1200, transparent=True)
    return elems, contps, max_element_id, max_contp_id, face_id, (idx_polys, idx_contps_reverse), max_deviation,(min_x, max_x)


def pixel_to_mortar_gmsh(all_elems, contps, idx_tuple, save_dir=None, start_index=1000, contp_start_index=1000, contact_type_dict={"type_name": "friction"}, face_start_index=0, mesh_size=10, split_stone=False):
    # find bounding box
    bounds = []
    # initialize a large value
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    stones = dict()
    bounds_low = []
    bounds_high = []
    for key, element in all_elems.items():
        if element.type.startswith('stone'):
            stones[key] = element

        else:
            min_x = min(min_x, np.min(np.asarray(element.vertices)[:, 0]))
            min_y = min(min_y, np.min(np.asarray(element.vertices)[:, 1]))
            max_x = max(max_x, np.max(np.asarray(element.vertices)[:, 0]))
            max_y = max(max_y, np.max(np.asarray(element.vertices)[:, 1]))

            if element.type == 'ground':
                bounds_low.append(element)
            elif element.type == 'beam':
                bounds_high.append(element)

    Z = 0
    mesh_size = mesh_size

    gmsh.initialize()
    # SetFactory("OpenCASCADE");
    gmsh.model.add("mortar")
    pointID = 1
    lineID = 1
    curveLoopID = 1
    curve_tags = []
    surface_ID = 1
    for key, value in stones.items():
        coords = value.vertices
        # create points
        point_tags = []
        for i in range(len(coords)):
            gmsh.model.geo.addPoint(
                coords[i][0], coords[i][1], Z, mesh_size, pointID)
            # if DEBUG:
            #     plt.scatter(coords[i][0], coords[i][1])
            point_tags.append(pointID)
            pointID += 1
        # create lines
        line_tags = []
        plot_stone = False
        for i in range(len(point_tags)):
            gmsh.model.geo.addLine(point_tags[i-1], point_tags[i], lineID)
            if DEBUG and (lineID == 641 or lineID == 1640 or plot_stone ==True):
                plt.plot([coords[i-1][0], coords[i][0]],
                         [coords[i-1][1], coords[i][1]])
                print(coords[i-1], coords[i])
                plot_stone = True
            line_tags.append(lineID)
            lineID += 1
        if DEBUG:
            plt.show()
        

        # create curve loop for each stone
        gmsh.model.geo.addCurveLoop(line_tags, curveLoopID)
        gmsh.model.addPhysicalGroup(1, line_tags, name=f"stone_{key}")
        curve_tags.append(curveLoopID)
        if split_stone:
            gmsh.model.geo.addPlaneSurface([curveLoopID], surface_ID)
            surface_ID += 1
        curveLoopID += 1

    # create surfaces -> create bounding box curve, add with stone curves
    if len(bounds_low) != 1:
        raise ValueError("There should be only one ground element")
    if len(bounds_high) != 1:
        raise ValueError("There should be only one beam element")
    # add groud line

    coords = bounds_low[0].vertices
    # sort coords by x coordinate
    coords = sorted(coords, key=lambda x: x[0])
    # create points
    point_tags = []
    min_x_min_y_poin_id = pointID
    for i in range(len(coords)):
        gmsh.model.geo.addPoint(
            coords[i][0], coords[i][1], Z, mesh_size, pointID)
        # if DEBUG:
        #     plt.scatter(coords[i][0], coords[i][1])
        point_tags.append(pointID)
        pointID += 1
    max_x_min_y_poin_id = pointID-1
    # create lines
    line_tags = []
    gound_line_tags = []
    for i in range(1, len(point_tags)):
        gmsh.model.geo.addLine(point_tags[i-1], point_tags[i], lineID)
        if DEBUG and (lineID == 1189 or lineID == 1191 or lineID == 1194):
            plt.plot([coords[i-1][0], coords[i][0]],
                     [coords[i-1][1], coords[i][1]])
        line_tags.append(lineID)
        gound_line_tags.append(lineID)
        lineID += 1
    gmsh.model.addPhysicalGroup(1, gound_line_tags, name="ground")
    # top bound
    coords_high = bounds_high[0].vertices
    # sort coords by x coordinate from large to small
    coords_high = sorted(coords_high, key=lambda x: x[0], reverse=True)
    point_tags = []
    gmsh.model.geo.addPoint(
        coords_high[0][0], coords_high[0][1], Z, mesh_size, pointID)
    max_x_max_y_poin_id = pointID
    # if DEBUG:
    #     plt.scatter(coords_high[0][0], coords_high[0][1])
    point_tags.append(pointID)
    pointID += 1
    # right bound
    gmsh.model.geo.addLine(max_x_min_y_poin_id, max_x_max_y_poin_id, lineID)
    if DEBUG and (lineID == 1189 or lineID == 1191 or lineID == 1194):
        plt.plot([coords[-1][0], coords_high[0][0]],
                 [coords[-1][1], coords_high[0][1]])
    line_tags.append(lineID)
    lineID += 1
    # create points

    for i in range(1, len(coords_high)):
        gmsh.model.geo.addPoint(
            coords_high[i][0], coords_high[i][1], Z, mesh_size, pointID)
        # if DEBUG:
        #     plt.scatter(coords_high[i][0], coords_high[i][1])
        point_tags.append(pointID)
        pointID += 1
    min_x_max_y_poin_id = pointID-1
    # create lines
    beam_line_tags = []
    for i in range(1, len(point_tags)):
        gmsh.model.geo.addLine(point_tags[i-1], point_tags[i], lineID)
        if DEBUG and (lineID == 1189 or lineID == 1191 or lineID == 1194):
            plt.plot([coords_high[i-1][0], coords_high[i][0]],
                     [coords_high[i-1][1], coords_high[i][1]])
        line_tags.append(lineID)
        beam_line_tags.append(lineID)
        lineID += 1
    gmsh.model.addPhysicalGroup(1, beam_line_tags, name="beam")
    # left bound
    gmsh.model.geo.addLine(min_x_max_y_poin_id, min_x_min_y_poin_id, lineID)
    if DEBUG and (lineID == 1189 or lineID == 1191 or lineID == 1194):
        plt.plot([coords_high[-1][0], coords[0][0]],
                 [coords_high[-1][1], coords[0][1]])
    line_tags.append(lineID)
    lineID += 1
    # create curve loop for bounds
    if DEBUG:
        plt.show()
    gmsh.model.geo.addCurveLoop(line_tags, curveLoopID)
    curve_tags.append(curveLoopID)
    curveLoopID += 1
    #gmsh.model.geo.addPoint(min_x, min_y, Z, mesh_size, pointID)
    #plt.scatter(min_x, min_y)
    #pointID += 1
    # gmsh.model.geo.addPoint(max_x, min_y, Z, mesh_size, pointID)
    # plt.scatter(max_x, min_y)
    # pointID += 1
    # gmsh.model.geo.addPoint(max_x, max_y, Z, mesh_size, pointID)
    # plt.scatter(max_x, max_y)
    # pointID += 1
    # gmsh.model.geo.addPoint(min_x, max_y, Z, mesh_size, pointID)
    # plt.scatter(min_x, max_y)
    # pointID += 1
    # gmsh.model.geo.addLine(pointID-4, pointID-1, lineID)
    # lineID += 1
    # gmsh.model.geo.addLine(pointID-3, pointID-4, lineID)
    # lineID += 1
    # gmsh.model.geo.addLine(pointID-2, pointID-3, lineID)
    # lineID += 1
    # gmsh.model.geo.addLine(pointID-1, pointID-2, lineID)
    # lineID += 1
    # gmsh.model.geo.addCurveLoop(
    #     [lineID-4, lineID-3, lineID-2, lineID-1], curveLoopID)
    # curve_tags.append(curveLoopID)
    # curveLoopID += 1
    gmsh.model.geo.addPlaneSurface(curve_tags, surface_ID)
    if DEBUG:
        plt.show()

    gmsh.model.geo.synchronize()
    # create mesh
    #gmsh.option.setNumber('Mesh.MeshSizeMin', 100)
    gmsh.model.mesh.setAlgorithm(dim=2, tag=1, val=5)
    gmsh.model.mesh.setSizeFromBoundary(dim=2, tag=1, val=10)
    gmsh.model.mesh.generate(2)

    if split_stone == True:
        # clear elements
        new_elems = {
            GROUND_ID_IN_MODEL: all_elems[GROUND_ID_IN_MODEL], BEAM_ID_IN_MODEL: all_elems[BEAM_ID_IN_MODEL]}
        ################################
        ### Save mesh per surface ######
        s_tags = []
        for e in gmsh.model.getEntities(dim=2):
            s_tag = e[1]
            s_tags.append(s_tag)
            tags, coord, param = gmsh.model.mesh.getNodes(2, s_tag, True)
            points = np.array(coord).reshape(-1, 3)
            tags = np.array(tags)
            elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(
                2, s_tag)
            # Renumbering nodes, new number of one point is the index of its tag in tags
            elemNodeTags_reordered = np.zeros((len(elemNodeTags),))-1
            for i_nt, node_tag in enumerate(tags):
                elemNodeTags_reordered = np.where(
                    elemNodeTags == node_tag, i_nt, elemNodeTags_reordered)
            cells = [
                ("triangle", (np.array(elemNodeTags_reordered).reshape(-1, 3)).astype(np.int64))
            ]
            if s_tag == surface_ID:
                element_type = 'mortar'
                mesh_name = '/mortar'
            else:
                element_type = f'stone_{s_tag}'
                mesh_name = '/stone_'+f"{s_tag}"
            meshio.write_points_cells(
                save_dir+mesh_name+".msh",
                points,
                cells)

            mesh = meshio.read(save_dir+mesh_name+".msh")
            new_elems, contps, idx_tuple, contp_start_index, face_start_index, start_index = \
                generate_block_from_mesh(mesh, new_elems, contps, idx_tuple, contact_type_dict, type=element_type,
                                         contp_start_index=contp_start_index, face_start_index=face_start_index, elem_start_index=start_index, max_y=max_y)

        # add contact points on ground and beam element
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)
        gmsh.write(save_dir+"/stone_ground_beam.msh")
        gmsh.finalize()
        contp_index = contp_start_index
        _conttype = ContType(
            contact_type_dict['type_name'], contact_type_dict)
        stone_ground_beam_bound_mesh = meshio.read(
            save_dir+"/stone_ground_beam.msh")
        physical_tags = stone_ground_beam_bound_mesh.cell_data['gmsh:physical']
        lines = stone_ground_beam_bound_mesh.cells_dict['line']

        for key, value in stone_ground_beam_bound_mesh.field_data.items():
            if key.startswith("stone"):
                continue
            elif key == 'ground':
                cand_id = GROUND_ID_IN_MODEL
            elif key == 'beam':
                cand_id = BEAM_ID_IN_MODEL
            else:
                raise ValueError("unknown key to identify element boundary")
            center = np.asarray(new_elems[cand_id].center)
            if value[1] != 1:
                raise ValueError(
                    "the boundary dimension of element should be 1")
            physical_tag = value[0]
            element_lines = lines[np.squeeze(
                np.argwhere(physical_tags[0] == physical_tag))]
            current_element_contps = []
            for line in element_lines:
                p_start = np.asarray(
                    [stone_ground_beam_bound_mesh.points[line[0]][0], stone_ground_beam_bound_mesh.points[line[0]][1]])
                p_end = np.asarray(
                    [stone_ground_beam_bound_mesh.points[line[1]][0], stone_ground_beam_bound_mesh.points[line[1]][1]])

                tangent = (p_start-p_end)/np.linalg.norm(
                    p_start-p_end)
                normal_3d = np.cross(np.asarray([
                    tangent[0], tangent[1], 0]), np.asarray([0, 0, 1]))
                normal = normal_3d[0:2]/np.linalg.norm(normal_3d[0:2])
                orient = np.asarray(
                    [-p_start[0]+center[0], -p_start[1]+center[1]])
                if np.dot(orient, normal) < 0:
                    normal = -normal

                _v_shift = 0.01
                shifte_v = move_one_point_to_another(
                    p_start, p_end, shift=_v_shift).tolist()

                contps[contp_index] = ContPoint(contp_index, shifte_v, cand_id, -1, tangent.tolist(), [
                                                0, 0], normal.tolist(), _conttype)
                current_element_contps.append(contp_index)
                contps[contp_index].section_h = np.linalg.norm(
                    np.asarray(p_start)-np.asarray(p_end))-2*_v_shift
                contps[contp_index].lever = 0.5*(np.linalg.norm(
                    np.asarray(p_start)-np.asarray(p_end))-2*_v_shift)
                idx_tuple[1].insert(
                    contp_index, (shifte_v[0], shifte_v[0], shifte_v[1], shifte_v[1], -normal[0], -normal[0], -normal[1], -normal[1], contps[contp_index].section_h, contps[contp_index].section_h))
                contp_index += 1

                shifte_v = move_one_point_to_another(
                    p_end, p_start, shift=_v_shift).tolist()
                contps[contp_index] = ContPoint(contp_index, shifte_v, cand_id, -1, tangent.tolist(), [
                                                0, 0], normal.tolist(), _conttype)
                current_element_contps.append(contp_index)
                contps[contp_index].section_h = np.linalg.norm(
                    np.asarray(p_start)-np.asarray(p_end))-2*_v_shift
                contps[contp_index].lever = 0.5*(np.linalg.norm(
                    np.asarray(p_start)-np.asarray(p_end))-2*_v_shift)
                idx_tuple[1].insert(
                    contp_index, (shifte_v[0], shifte_v[0], shifte_v[1], shifte_v[1], -normal[0], -normal[0], -normal[1], -normal[1], contps[contp_index].section_h, contps[contp_index].section_h))
                contp_index += 1
            new_elems[cand_id].contps = current_element_contps

        ### End save mesh per surface ######
        ####################################

    elif split_stone == False:
        #######################################
        ### Use Gmsh to save mortar mesh ######
        # gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 0)
        # gmsh.model.mesh.recombine()
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)
        gmsh.write(save_dir+"/stone_ground_beam.msh")
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(save_dir+"/mortar.msh")
        gmsh.finalize()

        mesh = meshio.read(save_dir+"/mortar.msh")

        i = 0
        contp_index = contp_start_index
        face_id = face_start_index
        for i_e, e in enumerate(mesh.cells_dict["triangle"]):
            # read the traingle meshes
            connect = []
            for n in range(len(e)):
                connect.append([
                    mesh.points[e[n]][0], mesh.points[e[n]][1]])
            center = cal_center(connect)
            # calculate mass
            area = cal_tri_area(connect)
            mass = area
            element_id = start_index+i_e
            # if center[1] > (max_y-3):
            #     new_type = 'ground'
            # else:
            #     new_type = 'mortar'
            new_type = 'mortar'
            shape_file = f"element_{element_id}.json"
            all_elems[element_id] = Element(
                element_id, center, mass, connect, type=new_type, shape_file=shape_file)
            current_element_contps = []
            for i_v, v in enumerate(connect):
                _conttype = ContType(
                    contact_type_dict['type_name'], contact_type_dict)
                v_prev = connect[i_v-1]
                v_next = connect[i_v+1] if i_v < len(connect)-1 else connect[0]
                for v_neighbor in [v_prev, v_next]:
                    tangent = (np.asarray(v_neighbor)-np.asarray(v))/np.linalg.norm(
                        np.asarray(v_neighbor)-np.asarray(v))
                    normal_3d = np.cross(np.asarray([
                        tangent[0], tangent[1], 0]), np.asarray([0, 0, 1]))
                    normal = normal_3d[0:2]/np.linalg.norm(normal_3d[0:2])
                    orient = np.asarray([-v[0]+center[0], -v[1]+center[1]])
                    if np.dot(orient, normal) < 0:
                        normal = -normal

                    _v_shift = 0.01
                    shifte_v = move_one_point_to_another(np.asarray(
                        v), np.asarray(v_neighbor), shift=_v_shift).tolist()
                    contps[contp_index] = ContPoint(contp_index, shifte_v, element_id, -1, tangent.tolist(), [
                                                    0, 0], normal.tolist(), _conttype)
                    current_element_contps.append(contp_index)

                    if contp_index % 2 == 0:
                        contps[contp_index].faceID = face_id
                    else:
                        face_id += 1
                        contps[contp_index].faceID = face_id
                    contps[contp_index].section_h = np.linalg.norm(
                        np.asarray(v_neighbor)-np.asarray(v))-2*_v_shift
                    contps[contp_index].lever = 0.5*(np.linalg.norm(
                        np.asarray(v_neighbor)-np.asarray(v))-2*_v_shift)
                    idx_tuple[1].insert(
                        contp_index, (shifte_v[0], shifte_v[0], shifte_v[1], shifte_v[1], -normal[0], -normal[0], -normal[1], -normal[1], contps[contp_index].section_h, contps[contp_index].section_h))
                    contp_index += 1
                all_elems[element_id].contps = current_element_contps

        stone_ground_beam_bound_mesh = meshio.read(
            save_dir+"/stone_ground_beam.msh")
        physical_tags = stone_ground_beam_bound_mesh.cell_data['gmsh:physical']
        lines = stone_ground_beam_bound_mesh.cells_dict['line']

        for key, value in stone_ground_beam_bound_mesh.field_data.items():
            if key.startswith("stone"):
                cand_id = int(key.split("_")[1])
            elif key == 'ground':
                cand_id = GROUND_ID_IN_MODEL
            elif key == 'beam':
                cand_id = BEAM_ID_IN_MODEL
            else:
                raise ValueError("unknown key to identify element boundary")
            center = np.asarray(all_elems[cand_id].center)
            if value[1] != 1:
                raise ValueError(
                    "the boundary dimension of element should be 1")
            physical_tag = value[0]
            element_lines = lines[np.squeeze(
                np.argwhere(physical_tags[0] == physical_tag))]
            current_element_contps = []
            if len(element_lines.shape) == 1:
                element_lines = np.expand_dims(element_lines, axis=0)
            for line in element_lines:
                p_start = np.asarray(
                    [stone_ground_beam_bound_mesh.points[line[0]][0], stone_ground_beam_bound_mesh.points[line[0]][1]])
                p_end = np.asarray(
                    [stone_ground_beam_bound_mesh.points[line[1]][0], stone_ground_beam_bound_mesh.points[line[1]][1]])

                tangent = (p_start-p_end)/np.linalg.norm(
                    p_start-p_end)
                normal_3d = np.cross(np.asarray([
                    tangent[0], tangent[1], 0]), np.asarray([0, 0, 1]))
                normal = normal_3d[0:2]/np.linalg.norm(normal_3d[0:2])
                orient = np.asarray(
                    [-p_start[0]+center[0], -p_start[1]+center[1]])
                if np.dot(orient, normal) < 0:
                    normal = -normal

                _v_shift = 0.01
                shifte_v = move_one_point_to_another(
                    p_start, p_end, shift=_v_shift).tolist()

                contps[contp_index] = ContPoint(contp_index, shifte_v, cand_id, -1, tangent.tolist(), [
                                                0, 0], normal.tolist(), _conttype)
                current_element_contps.append(contp_index)
                contps[contp_index].section_h = np.linalg.norm(
                    np.asarray(p_start)-np.asarray(p_end))-2*_v_shift
                contps[contp_index].lever = 0.5*(np.linalg.norm(
                    np.asarray(p_start)-np.asarray(p_end))-2*_v_shift)
                idx_tuple[1].insert(
                    contp_index, (shifte_v[0], shifte_v[0], shifte_v[1], shifte_v[1], -normal[0], -normal[0], -normal[1], -normal[1], contps[contp_index].section_h, contps[contp_index].section_h))
                contp_index += 1

                shifte_v = move_one_point_to_another(
                    p_end, p_start, shift=_v_shift).tolist()
                contps[contp_index] = ContPoint(contp_index, shifte_v, cand_id, -1, tangent.tolist(), [
                                                0, 0], normal.tolist(), _conttype)
                current_element_contps.append(contp_index)
                contps[contp_index].section_h = np.linalg.norm(
                    np.asarray(p_start)-np.asarray(p_end))-2*_v_shift
                contps[contp_index].lever = 0.5*(np.linalg.norm(
                    np.asarray(p_start)-np.asarray(p_end))-2*_v_shift)
                idx_tuple[1].insert(
                    contp_index, (shifte_v[0], shifte_v[0], shifte_v[1], shifte_v[1], -normal[0], -normal[0], -normal[1], -normal[1], contps[contp_index].section_h, contps[contp_index].section_h))
                contp_index += 1
            all_elems[cand_id].contps = current_element_contps
        new_elems = all_elems.copy()
        ### End use gmsh to save mortar mesh ###
        ########################################
    return new_elems, contps, idx_tuple


def reset_face_id(contps, idx_normal_cand_points):
    face_id = 0
    for key, value in contps.copy().items():
        same_point_ids = list(idx_normal_cand_points.nearest(
            (value.normal[0], value.normal[0], value.normal[1], value.normal[1], value.cand, value.cand, value.anta, value.anta), 2))
        if len(same_point_ids) != 2:
            raise ValueError("There should be two points on one face")
        contps[same_point_ids[0]].faceID = face_id
        contps[same_point_ids[1]].faceID = face_id
        face_id += 1
    return contps

def scale_image(_image_name, data_config=None):
    global BEAM_ID_IN_MATRIX
    global GROUND_ID_IN_MATRIX
    # *******************************************
    # ******** Read image in data folder *******
    # *******************************************
    # if _image_name is path to an image
    if isinstance(_image_name, str):
        img_ori = cv2.imread(
            _data_dir+'/'+_image_name, cv2.IMREAD_GRAYSCALE)
    

    else:
        raise ValueError('Image name is not valid')

    # img_ori = img_ori[0:40,:]
    if 'img_scale_method' not in data_config.keys() or 'img_processing' not in data_config.keys() or data_config['img_processing'] == False:
        pass
    else:
        if bool(data_config['binarilize']):
            img_ori = 255.0*(img_ori > int(data_config['binary_threshold']))
        if bool(data_config['closing']):
            img_ori = 255.0*skimage.morphology.binary_closing(
                img_ori, footprint=None, out=None)
    img_ori_s_m_ratio = 1-(len(np.argwhere(img_ori != 0)) /
                           img_ori.shape[0]/img_ori.shape[1])
    # *******************************************
    # ******** Resize image **********************
    # *******************************************
    if 'img_scale_method' not in data_config.keys() or data_config['img_scale_method'] == 'simple':
        scale = img_ori.shape[0]/int(data_config['scale_img_to_height'])
        scale = 1
        # binarilize
        img_bi = 1.0*(img_ori == 255)
        # # add ground and beam by adding four rows
        # img_beam = np.vstack(
        #     (img_bi, 2*np.ones((2*int(scale), img_bi.shape[1]))))
        # img_beam_ground = np.vstack(
        #     (3*np.ones((2*int(scale), img_bi.shape[1])), img_beam))
        img_labeled = skimage.measure.label(
            img_bi, background=0, connectivity=2)
        _max_label = np.max(img_labeled)
        img_resized = np.zeros((int(data_config['scale_img_to_height']),
                                int(data_config['scale_img_to_width'])))
        for region in regionprops(img_labeled):
            if region.area > data_config['minimum_stone_size']:
                region_labeled = np.where(
                    img_labeled == region.label, region.label, 0)
                region_labeled_resized = cv2.resize(region_labeled, (int(data_config['scale_img_to_width']),
                                int(data_config['scale_img_to_height'])),                                                    interpolation=cv2.INTER_NEAREST)
                #region_labeled_resized = region_labeled
                img_resized = np.where((img_resized == 0) & (
                    region_labeled_resized != 0), region_labeled_resized, img_resized)
        # add write pixels to four boundaries
        if 'add_bound' not in data_config.keys() or bool(data_config['add_bound']) == True:
            img_resized = np.vstack(
                (img_resized, np.zeros((2, img_resized.shape[1]))))
            img_resized = np.vstack(
                (np.zeros((2, img_resized.shape[1])), img_resized))
            img_resized = np.hstack(
                (img_resized, np.zeros((img_resized.shape[0], 2))))
            img_resized = np.hstack(
                (np.zeros((img_resized.shape[0], 2)), img_resized))
        img_resized_s_m_ratio = len(np.argwhere(
            img_resized != 0))/img_resized.shape[0]/img_resized.shape[1]
        if 'img_has_ground_beam' not in data_config.keys() or bool(data_config['img_has_ground_beam']) == False:
            # add ground and beam by adding four rows
            img_beam = np.vstack(
                ((_max_label+1)*np.ones((2, img_resized.shape[1])), img_resized))
            #global BEAM_ID_IN_MATRIX
            BEAM_ID_IN_MATRIX = _max_label+1
            img_beam_ground = np.vstack(
                (img_beam, (_max_label+2)*np.ones((2, img_beam.shape[1]))))
            #global GROUND_ID_IN_MATRIX
            GROUND_ID_IN_MATRIX = _max_label+2
            img = img_beam_ground
        else:
            img_beam_ground = img_resized
            BEAM_ID_IN_MATRIX = np.max(img_labeled[0, :])
            GROUND_ID_IN_MATRIX = np.max(img_labeled[-1, :])
            img = img_beam_ground

        kmodel_info = {
            "img_width": img_beam_ground.shape[1], "img_height": img_beam_ground.shape[0], "scale": scale, "original_stone_mortar_ratio": img_ori_s_m_ratio, "scaled_stone_mortar_ratio": img_resized_s_m_ratio}
    return img, kmodel_info

def create_kmodel_polystone(_image_name, result_directory, data_config=None, meshing=True):
    set_dimension(2)
    if not os.path.exists(result_directory+'/'+_image_name.split('.')[0]):
        os.system('mkdir '+result_directory+'/'+_image_name.split('.')[0])
    kmodel_save_dir = result_directory + \
        '/'+_image_name.split('.')[0]+'/kmodel'
    if os.path.exists(kmodel_save_dir+'/'):
        os.system('rm -rf '+kmodel_save_dir)
    os.system('mkdir '+kmodel_save_dir)

    img, kmodel_info = scale_image(_image_name, data_config=data_config)
    contact_type_dict = {
        "type_name": data_config['initial_contact_type'], "mu": float(data_config['mu'])}
    if 'split_stone' not in data_config.keys():
        split_stone = False
    else:
        split_stone = bool(data_config['split_stone'])
    if 'downsample_method' not in data_config.keys() or 'number_segments' not in data_config.keys():
        _downsample_method = 'voxel'
        _number_segments = 10
    else:
        _downsample_method = data_config['downsample_method']
        _number_segments = data_config['number_segments']
    stone_gb_elems, contps, max_element_id, max_contp_id, max_face_id, (idx, idx_reverse), deviation,xbounds = pixel_to_elem_polystone(
        img.astype(np.uint16), contp_start_index=0, face_start_index=0, contact_type_dict=contact_type_dict, downsample_method=_downsample_method, point_grid_size=data_config['point_grid_size'], number_segments=_number_segments, _result_directory=kmodel_save_dir)
    if meshing:
        elems, contps, (idx, idx_reverse) = pixel_to_mortar_gmsh(
            stone_gb_elems, contps, (idx, idx_reverse), save_dir=kmodel_save_dir, start_index=1+max_element_id, contp_start_index=1+max_contp_id, face_start_index=1+max_face_id, contact_type_dict=contact_type_dict, mesh_size=data_config['mesh_size'], split_stone=split_stone)
        # # remove elements beyond xbounds
        # removed_elements = []
        # for key, value in elems.copy().items():
        #     if value.center[0] < xbounds[0] or value.center[0] > xbounds[1]:
        #         removed_elements.append(key)
        #         del elems[key]
        # for key, value in contps.copy().items():
        #     if value.cand in removed_elements or value.anta in removed_elements:
        #         del contps[key]

        contps, idx_normal_cand_points = cal_anta_id_with_rtree(
            contps, (idx, idx_reverse))
        contps = reset_face_id(contps, idx_normal_cand_points)

        for element in elems.values():
            element.dl = [0, element.mass, 0]
            element.ll = [element.mass, 0, 0]
        #elems, contps=scale_model(elems, contps, data_config['scale_to_m_x'], data_config['scale_to_m_y'])
        write_to_csv(elems, contps, kmodel_save_dir)

        to_csv_2d(elems, contps, "model", kmodel_save_dir)
    else:
        elems = {}
        contps = {}
        idx_normal_cand_points = None

    cv2.imwrite(kmodel_save_dir+"/matrix_id.tiff",
                img.astype(np.float32)/np.max(img))
    cv2.imwrite(kmodel_save_dir+"/matrix_type.tiff",
                img.astype(np.float32)/np.max(img))
    cv2.imwrite(kmodel_save_dir+"/scaled_img.png",
                (img/np.max(img)*255).astype(np.uint8))

    # write information to txt
    with open(kmodel_save_dir+'/AA_info.txt', 'w+') as f:
        f.write('scale: '+str(kmodel_info['scale'])+'\n')
        f.write(f"Number of elements: {len(list(elems.items()))}\n")
        f.write(f"Number of contact points: {len(list(contps.items()))}\n")
        f.write(f"Stone/mortar ratio in original image" +
                str(kmodel_info['original_stone_mortar_ratio'])+"\n")
        f.write(f"Stone/mortar ratio in scaled image" +
                str(kmodel_info['scaled_stone_mortar_ratio'])+"\n")
        f.write(f"Deviation in kmodel" +
                str(deviation)+"\n")
    kmodel_info.update({"nb_elems": len(list(elems.items())),
                       "nb_contps": len(list(contps.items()))})
    return kmodel_save_dir, kmodel_info


def write_to_csv(elems, contps, _root_dir):
    # write element to csv
    id = []
    type = []
    cx = []
    cy = []
    mass = []
    shape = []
    dl_fx = []
    dl_fy = []
    dl_mz = []
    ll_fx = []
    ll_fy = []
    ll_mz = []
    import json
    for e in elems.values():
        id.append(e.id)
        type.append(e.type)
        cx.append(e.center[0])
        cy.append(e.center[1])
        mass.append(e.mass)
        _json_shape_file_name = os.path.join(_root_dir, f"element_{e.id}.json")
        with open(_json_shape_file_name, 'w') as f:
            json.dump(e.vertices, f, indent=6)
        shape.append(f"element_{e.id}.json")
        dl_fx.append(e.dl[0])
        dl_fy.append(e.dl[1])
        dl_mz.append(e.dl[2])
        ll_fx.append(e.ll[0])
        ll_fy.append(e.ll[1])
        ll_mz.append(e.ll[2])
    _csv_file_name = os.path.join(_root_dir, "element.csv")
    import pandas as pd

    df = pd.DataFrame({"id": id, "type": type, "cx": cx, "cy": cy, "mass": mass, "shape": shape,
                       "dl_fx": dl_fx, "dl_fy": dl_fy, "dl_mz": dl_mz, "ll_fx": ll_fx, "ll_fy": ll_fy, "ll_mz": ll_mz})
    df.to_csv(_csv_file_name, index=False, float_format='%.7f')
    # write contact point to csv
    id = []
    x = []
    y = []
    nx = []
    ny = []
    t1x = []
    t1y = []
    candidate_id = []
    antagonist_id = []
    section_height = []
    lever = []
    face_id = []
    counter_point = []
    contact_type = []
    mu = []
    for p in contps.values():
        id.append(p.id)
        x.append(p.coor[0])
        y.append(p.coor[1])
        nx.append(p.normal[0])
        ny.append(p.normal[1])
        t1x.append(p.tangent1[0])
        t1y.append(p.tangent1[1])
        candidate_id.append(p.cand)
        antagonist_id.append(p.anta)
        section_height.append(p.section_h)
        lever.append(p.lever)
        face_id.append(p.faceID)
        counter_point.append(p.counterPoint)
        contact_type.append(p.cont_type.type)
        mu.append(p.cont_type.mu)
    _csv_file_name = os.path.join(_root_dir, "point.csv")

    df = pd.DataFrame({"id": id, "x": x, "y": y, "nx": nx, "ny": ny, "t1x": t1x, "t1y": t1y, "candidate_id": candidate_id, "antagonist_id": antagonist_id,
                       "section_height": section_height, "lever": lever, "face_id": face_id, "counter_point": counter_point, "contact_type": contact_type, "mu": mu, "fc": 0, "ft": 0, "cohesion": 0})
    df.to_csv(_csv_file_name, index=False, float_format='%.7f')

def scale_model(elems, contps, scale_x,scale_y):
    for e in elems.values():
        e.center = [e.center[0]*scale_x, e.center[1]*scale_y]
        e.mass = e.mass*scale_x*scale_y
        for vtc in e.vertices:
            vtc[0] *= scale_x
            vtc[1] *= scale_y
    for p in contps.values():
        p.coor = [p.coor[0]*scale_x, p.coor[1]*scale_y]
        p.section_h *= scale_y
        p.lever *= scale_y
    return elems, contps

with open(_data_dir+'/config.json') as config_file:
    data_config = json.load(config_file)

if not os.path.exists(current_result_dir):
    os.system('mkdir -p '+current_result_dir)

with open(current_result_dir+'/config.json', 'w+') as f:
    json.dump(data_config, f, indent=4)

# generate model
kmodel_save_dir, kmodel_info = create_kmodel_polystone(
    imagename, current_result_dir, data_config=data_config)
