import laspy
import numpy as np
import os
import open3d as o3d
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
from tqdm import tqdm


def read_las_file(file_path):
    
    # read las file
    las_file = laspy.read(file_path)

    # get the point data
    point_data = las_file.points
    # get the x, y, z coordinates
    x = point_data.x
    y = point_data.y
    z = point_data.z
    # get the intensity values
    semantics = point_data.intensity

    # stock the x, y, z, semantics in a numpy array
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    semantics = np.array(semantics)

    # get color values from las file
    red = point_data.red
    green = point_data.green
    blue = point_data.blue

    colors = np.array([red, green, blue]).T

    # stack the arrays
    points_semantics_source = np.vstack((x, y, z, semantics)).T

    print(points_semantics_source.shape)

    # assert the length of the arrays
    assert len(x) == len(y) == len(z) == len(semantics), "Length of x, y, z, and intensity arrays do not match."

    # print the numbers of points and semantics
    print(f"Number of points: {len(x)}")
    print(f"Number of semantics: {np.unique(semantics).size}")

    return points_semantics_source, colors

def read_las_files(file_paths):
    """
    Read multiple LAS files and return their point data and colors.
    """
    semantics_N = 0
    points_semantics = []
    colors = []

    background_semantics = []

    for file_path in file_paths:
        points_semantics_source, colors_source = read_las_file(file_path)
        # update points_semantics_source with semantics_N
        points_semantics_source[:, 3] += semantics_N
        background_semantics.append(np.max(points_semantics_source[:, 3]))
        semantics_N = np.max(points_semantics_source[:, 3]) + 1
        points_semantics.append(points_semantics_source)
        colors.append(colors_source)

    # concatenate all points_semantics and colors
    points_semantics = np.concatenate(points_semantics, axis=0)
    colors = np.concatenate(colors, axis=0)

    # reset the background semantics
    for i in range(len(background_semantics)-1):
        points_semantics[points_semantics[:, 3] == background_semantics[i], 3] = background_semantics[-1]
    
    return points_semantics, colors


def compute_box_size(points_semantics, sem):
    points_sem = points_semantics[points_semantics[:, 3] == sem]
    min_coords = np.min(points_sem[:, :3], axis=0)
    max_coords = np.max(points_sem[:, :3], axis=0)
    box_size = np.prod(max_coords - min_coords)
    return (sem, box_size)

def sort_semantics_by_box_size(points_semantics, n_jobs=-1):
    """
    Sort semantics by the size of the bounding box using joblib parallel processing and tqdm.

    Parameters:
        points_semantics (np.ndarray): Array of shape (N, 4) with XYZ and semantic label.
        n_jobs (int): Number of parallel jobs (-1 uses all cores).
    
    Returns:
        np.ndarray: Updated points_semantics with remapped semantic labels sorted by box size.
    """
    semantics = points_semantics[:, 3]
    unique_semantics = np.unique(semantics)

    # Manually build the task list for tqdm to monitor
    tasks = (delayed(compute_box_size)(points_semantics, sem) for sem in unique_semantics)
    box_sizes = Parallel(n_jobs=n_jobs)(
        tqdm(tasks, total=len(unique_semantics), desc="Computing box sizes")
    )

    # Sort and remap semantics
    sorted_box_sizes = sorted(box_sizes, key=lambda x: x[1], reverse=True)
    sorted_semantics = [sem for sem, _ in sorted_box_sizes]
    semantics_mapping = {sem: i for i, sem in enumerate(sorted_semantics)}
    points_semantics[:, 3] = np.vectorize(semantics_mapping.get)(points_semantics[:, 3])

    return points_semantics


def save_points_to_las(points, color, filename):
    # Create a new LAS header and file
    header = laspy.LasHeader(point_format=3, version="1.2")
    las_file = laspy.LasData(header)

    # Set coordinates
    las_file.x = points[:, 0]
    las_file.y = points[:, 1]
    las_file.z = points[:, 2]

    # Handle intensity (semantics)
    semantics = points[:, 3].astype(np.int16)  # Promote to signed int
    max_intensity = semantics[semantics != -1].max()

    if len(semantics[semantics == max_intensity]) < len(semantics)/10:
        semantics[semantics == -1] = max_intensity + 1 
        print(f"Max intensity: {max_intensity + 1}")
    else:
        semantics[semantics == -1] = max_intensity 
        print(f"Max intensity: {max_intensity}")
    las_file.intensity = semantics.astype(np.uint16)  # Cast back to uint16

    

    # Set RGB color
    las_file.red = color[:, 0].astype(np.uint16)
    las_file.green = color[:, 1].astype(np.uint16)
    las_file.blue = color[:, 2].astype(np.uint16)

    # Write the LAS file
    las_file.write(filename)
