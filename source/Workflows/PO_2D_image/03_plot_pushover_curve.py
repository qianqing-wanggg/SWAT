import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString,Point
import glob
plt.style.use('science')
#set plt size
plt.rcParams["figure.figsize"] = (2,1.5)

#discrete color map
cmap = plt.get_cmap('tab20')
colors = [
    "#045275",
    "#089099",
    "#7CCBA2",
    "#FCDE9C",
    "#F0746E",
    "#DC3977",
    "#7C1D6F",
    "#EEB479"
]

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

typename = imagename.split(".")[0]
pushover_elastic_dir = 'run_pushover'#
_result_directory = current_result_dir+'/'+typename


def plot_elastic_and_plastic_curve():
    fix, ax = plt.subplots()

    intersections_force = []
    intersections_displacement = []
    ultimate_drifts = []

    first_line = None
    if os.path.isdir(_result_directory+'/'+pushover_elastic_dir+'/'+typename+'_sample0'):
        #read force_displacement.txt in the directory as numpy array
        if os.path.exists(_result_directory+'/'+pushover_elastic_dir+'/'+typename+'_sample0'+'/force_displacement.txt'):
            # read by line
            fs = []
            ds = []
            result_file_path = _result_directory+'/'+pushover_elastic_dir+'/'+typename+'_sample0'+'/force_displacement.txt'
            with open(result_file_path) as f:
                next(f)  # skip the first line
                for line in f:
                    # split the line into a list of items
                    items = line.split(',')
                    # convert items to float
                    items = [float(item) for item in items]
                    # append the list of items to the list of data
                    fs.append(items[0])
                    ds.append(items[1]*1000)

            cumds = []
            for i in range(len(ds)):
                if i>0:
                    cumds.append(ds[i]+cumds[-1])
                else:
                    cumds.append(ds[i])
            cumds = [cumd for cumd in cumds]
            fs = np.array(fs)
            cumds = np.array(cumds)
            ax.plot(cumds, fs, '-',color=colors[-1],alpha = 1)

    # add grid line at every 10 in y
    plt.grid(axis='y', which='major', color='#CCCCCC', linestyle='--')
    plt.grid(axis='x', which='major', color='#CCCCCC', linestyle='--')
    ax.set_xlabel('Displacement (mm)')
    ax.set_ylabel('Stress (MPa)')
    plt.savefig(_result_directory+f"/pushover_curve.png", dpi=600, transparent=False)

    return intersections_force,intersections_displacement,ultimate_drifts

plot_elastic_and_plastic_curve()
