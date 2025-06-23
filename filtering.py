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
from itertools import groupby
from operator import itemgetter
from scipy.spatial import ConvexHull



def sor_filtering(points_semantics, n_neighbors=10, std_ratio=1.0):
    """
    Apply Statistical Outlier Removal (SOR) filtering to the point cloud.
    Points identified as outliers will be relabeled as -1.
    """
    def sor_filtering_single(xyz, semantic_id, n_neighbors=10, std_ratio=1.0):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.copy())  # FIXED: Make array writable

        # Apply SOR
        _, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=n_neighbors, std_ratio=std_ratio
        )

        inlier_mask = np.zeros(len(xyz), dtype=bool)
        inlier_mask[inlier_indices] = True
        return semantic_id, inlier_mask
    
    def build_semantic_maps(points_semantics):
        """
        Efficiently build maps from semantic ID to xyz and to indices.
        Returns:
            - id_to_xyz: {semantic_id: xyz array}
            - id_to_indices: {semantic_id: indices array}
        """
        from collections import defaultdict

        id_to_xyz = defaultdict(list)
        id_to_indices = defaultdict(list)

        for idx, point in enumerate(points_semantics):
            sem = int(point[3])
            id_to_xyz[sem].append(point[:3])
            id_to_indices[sem].append(idx)

        # Convert lists to arrays
        id_to_xyz = {k: np.array(v) for k, v in id_to_xyz.items()}
        id_to_indices = {k: np.array(v, dtype=int) for k, v in id_to_indices.items()}

        return id_to_xyz, id_to_indices

    semantics_ids = np.unique(points_semantics[:, 3])
    id_to_xyz, id_to_indices = build_semantic_maps(points_semantics)

    results = Parallel(n_jobs=8)(
        delayed(sor_filtering_single)(id_to_xyz[sem], sem, n_neighbors, std_ratio)
        for sem in tqdm(semantics_ids, desc="SOR Filtering")
    )

    for sem_id, inlier_mask in results:
        original_indices = id_to_indices[sem_id]
        outlier_indices = original_indices[~inlier_mask]
        points_semantics[outlier_indices, 3] = -1

    return points_semantics

def compute_convex_hull_density(points):
    """
    Compute point cloud density as: Number of points / Convex hull volume.

    Parameters:
        points (np.ndarray): Array of shape (N, 3) representing XYZ coordinates.

    Returns:
        density (float): Point density (points per unit volume).
        volume (float): Convex hull volume.
    """
    if points.shape[0] < 4:
        # At least 4 non-coplanar points are needed for a 3D hull
        return 0.0, 0.0

    try:
        hull = ConvexHull(points)
        volume = hull.volume
        if volume > 0:
            density = points.shape[0] / volume
        else:
            density = 0.0
        return density, volume
    except Exception as e:
        print(f"Convex hull error: {e}")
        return 0.0, 0.0

def calculate_density(points_semantics, radius=0.1, n_jobs=-1):
    def compute_point_density(xyz, semantics_id, radius=0.1):
        # if the number of points is greater than 1000000, this indicates that the point cloud is background 
        # and we can skip the density check
        if len(xyz) > 500000:
            return semantics_id, 0  # Return high density and zero variance for background
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.copy())  # <== MAKE IT WRITEABLE

        # KDTree for neighbor search
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        densities = []

        for i in range(len(xyz)):
            k, _, _ = pcd_tree.search_radius_vector_3d(xyz[i], radius)
            densities.append(k)

        avg_density = np.mean(densities)

        return semantics_id, avg_density
    
    semantics_ids = np.unique(points_semantics[:, 3])
    
    # Pre-filter points for each semantic label
    id_to_xyz = group_xyz_by_semantics(points_semantics)

    print("Starting density filtering...")
    # Parallel density checks with tqdm
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_point_density)(id_to_xyz[sid], sid, radius)
        for sid in tqdm(semantics_ids, desc="Filtering by density")
    )

    results = np.array(results)
    return results

def pc_filter(points_semantics, semantics_attribute, upper_bound_threshold=None, lower_bound_threshold=None):
    if upper_bound_threshold is not None:
        print(f"Unique semantics IDs before upper bound filtering: {len(np.unique(points_semantics[:, 3]))}")
        to_remove = {semantics_id for semantics_id, attribute in semantics_attribute if attribute > upper_bound_threshold}  # remove semantics IDs with attribute above the upper bound threshold
        # Vectorized mask: True for rows to be removed
        mask = np.isin(points_semantics[:, 3], list(to_remove))
        points_semantics[mask, 3] = -1
        print(f"Unique semantics IDs after filtering: {len(np.unique(points_semantics[:, 3]))}")

    if lower_bound_threshold is not None:
        print(f"Unique semantics IDs before lower bound filtering: {len(np.unique(points_semantics[:, 3]))}")
        to_remove = {semantics_id for semantics_id, attribute in semantics_attribute if attribute < lower_bound_threshold}
        # Vectorized mask: True for rows to be removed
        mask = np.isin(points_semantics[:, 3], list(to_remove))
        points_semantics[mask, 3] = -1
        print(f"Unique semantics IDs after filtering: {len(np.unique(points_semantics[:, 3]))}")

    return points_semantics

def group_xyz_by_semantics(points_semantics):
    # Sort by semantic label
    sorted_idx = np.argsort(points_semantics[:, 3])
    sorted_points = points_semantics[sorted_idx]
    
    # Extract unique labels and grouped xyz arrays
    semantics = sorted_points[:, 3]
    xyz = sorted_points[:, :3]

    id_to_xyz = {}
    for sem, group in groupby(zip(semantics, xyz), key=itemgetter(0)):
        id_to_xyz[sem] = np.array([x for _, x in group])
    
    return id_to_xyz


def cluster_filter(points_semantics, eps=0.5, min_samples=10, n_jobs=8):
    semantics_ids = np.unique(points_semantics[:, 3])

    def process_semantic_group(sem_id):
        # Find global indices for this semantic group
        group_indices = np.where(points_semantics[:, 3] == sem_id)[0]
        group_points = points_semantics[group_indices, :3]

        if len(group_points) < min_samples:
            return group_indices  # All treated as outliers

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(group_points)
        labels, counts = np.unique(clustering.labels_, return_counts=True)

        # Ignore noise-only cases (no valid clusters)
        if np.all(labels == -1):
            return group_indices

        largest_cluster_label = labels[np.argmax(counts)]
        inliers_local = np.where(clustering.labels_ == largest_cluster_label)[0]
        all_local = np.arange(len(group_indices))
        outliers_local = np.setdiff1d(all_local, inliers_local)

        # Return global indices of outliers
        return group_indices[outliers_local]

    # Parallel loop over semantic IDs
    outlier_indices_all = Parallel(n_jobs=n_jobs)(
        delayed(process_semantic_group)(sid) for sid in semantics_ids
    )

    all_outlier_indices = np.concatenate(outlier_indices_all)
    points_semantics[all_outlier_indices, 3] = -1

    # print the number of outliers
    print(f"Number of outliers from DBSCAN filter: {len(all_outlier_indices)}")

    return points_semantics



def size_filter(points_semantics, min_horizontal_length=0.5, max_horizontal_length=5.0, min_vertical_length=0.5, max_vertical_length=5.0):
    semantics_ids = np.unique(points_semantics[:, 3])

    def process_semantic_group(points_semantics, semantics_id, min_horizontal_length, max_horizontal_length, min_vertical_length, max_vertical_length):
        # Get indices of points with this semantics
        indices = np.where(points_semantics[:, 3] == semantics_id)[0]
        local_points_semantics = points_semantics[indices]
        xyz = local_points_semantics[:, :3]

        # Calculate the bounding box
        min_x, min_y, min_z = np.min(xyz, axis=0)
        max_x, max_y, max_z = np.max(xyz, axis=0)

        # Calculate lengths
        horizontal_length = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
        vertical_length = max_z - min_z

        # return True if the lengths are within the specified range; otherwise, return False
        if min_horizontal_length <= horizontal_length <= max_horizontal_length and min_vertical_length <= vertical_length <= max_vertical_length:
            return semantics_id, True
        else:
            return semantics_id, False
        
    # Parallel processing
    results = Parallel(n_jobs=8)(
        delayed(process_semantic_group)(points_semantics, sid, min_horizontal_length, max_horizontal_length, min_vertical_length, max_vertical_length)
        for sid in semantics_ids
    )

    # set the semantics to -1 for the points that do not pass the filter
    for semantics_id, keep in results:
        if not keep:
            indices = np.where(points_semantics[:, 3] == semantics_id)[0]
            points_semantics[indices, 3] = -1

    # print the number of semantics_ids that passed the filter
    print(f"Number of semantics_ids that passed the size filter: {len(np.unique(points_semantics[:, 3]))-1}")
    # print the number of semantics_ids that did not pass the filter
    print(f"Number of semantics_ids that did not pass the size filter: {len(semantics_ids) - len(np.unique(points_semantics[:, 3]))+1}")
    return points_semantics




def calculate_hwr(points_semantics, n_jobs=-1):
    def compute_point_hwr(xyz, semantics_id):
        # If it's too big (likely background), skip PCA
        if len(xyz) > 500000:
            return semantics_id, 0

        # Center the points
        xyz = xyz - np.mean(xyz, axis=0)

        pca = PCA(n_components=3)
        pca.fit(xyz)

        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_

        z_axis = np.array([0, 0, 1])
        cosines = np.abs(np.dot(eigenvectors, z_axis))
        index = np.argmax(cosines)

        other_indices = [0, 1, 2]
        other_indices.remove(index)

        width_index = min(other_indices, key=lambda i: eigenvalues[i])
        length_index = max(other_indices, key=lambda i: eigenvalues[i])

        pca_points = pca.transform(xyz)
        height = np.ptp(pca_points[:, index])
        width = np.ptp(pca_points[:, width_index])

        hwr = height / width if width > 1e-5 else np.inf
        return semantics_id, hwr

    semantics_ids = np.unique(points_semantics[:, 3])
    id_to_xyz = group_xyz_by_semantics(points_semantics)

    print("Starting HWR computation...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_point_hwr)(id_to_xyz[sid], sid)
        for sid in tqdm(semantics_ids, desc="Computing HWR")
    )

    results = np.array(results, dtype=object)
    # delete the row where hwr is 0 (likely background)
    results = results[results[:, 1] != 0]
    return results

def calculate_pca_curvature(points_semantics, n_jobs=-1):
    """
    Compute PCA-based curvature for each semantic segment.
    Curvature is defined as: λ3 / (λ1 + λ2 + λ3), where λi are PCA eigenvalues.
    
    :param points_semantics: numpy array of shape (N, 4)
    :param n_jobs: Number of parallel jobs
    :return: numpy array of (semantic_id, curvature)
    """
    def compute_curvature(xyz, semantics_id):
        if len(xyz) < 3:
            return semantics_id, 0.0

        xyz = xyz - np.mean(xyz, axis=0)
        pca = PCA(n_components=3)
        pca.fit(xyz)

        eigenvalues = np.sort(pca.explained_variance_)[::-1]  # λ1, λ2, λ3
        λ1, λ2, λ3 = eigenvalues

        total = λ1 + λ2 + λ3
        curvature = λ3 / total if total > 1e-10 else 0.0

        return semantics_id, curvature

    semantics_ids = np.unique(points_semantics[:, 3])
    id_to_xyz = group_xyz_by_semantics(points_semantics)

    print("Starting PCA curvature computation...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_curvature)(id_to_xyz[sid], sid)
        for sid in tqdm(semantics_ids, desc="Computing PCA curvature")
        if sid != -1
    )

    return np.array(results, dtype=object)