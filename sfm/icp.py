import open3d as o3d
import numpy as np
import laspy
import os
import pdal
import json


def load_las_to_o3d(file_path, fixed_center=None, downsample_voxel_size=None, load_colors=False):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T

    if fixed_center is not None:
        points -= fixed_center

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Load colors if requested and available
    if load_colors and hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        # Convert from uint16 [0,65535] to float [0,1] as expected by Open3D
        colors = np.vstack((las.red.astype(float) / 65535.0,
                           las.green.astype(float) / 65535.0,
                           las.blue.astype(float) / 65535.0)).T
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if downsample_voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)

    # print the range of coordinates
    print(f"Loaded {file_path} with {len(points)} points (colors: {pcd.has_colors()}).")
    return pcd

def save_o3d_to_las(pcd, file_path, fixed_center=None, save_colors=False):
    """
    Save an Open3D point cloud to a LAS file with millimeter-level accuracy.

    Parameters:
    - pcd: open3d.geometry.PointCloud
    - file_path: Output LAS file path
    - fixed_center: If provided, it will be added back to restore original coordinates
    - save_colors: If True, save color information if available
    """
    points = np.asarray(pcd.points)

    if fixed_center is not None:
        points += fixed_center

    # Determine point format based on available data and save_colors flag
    point_format = 0  # Default minimal format
    if save_colors and pcd.has_colors():
        point_format = 3  # Point format 3 includes RGB
    elif pcd.has_normals():
        point_format = 1  # Point format 1 has basic fields
    else:
        point_format = 0  # Point format 0 is minimal

    # Create LAS header with high precision
    header = laspy.LasHeader(point_format=point_format, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])  # 1 millimeter resolution
    header.offsets = np.min(points, axis=0)  # minimize coordinate compression error

    # Create and fill LAS data
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Add colors if requested and available
    if save_colors and pcd.has_colors():
        colors = np.asarray(pcd.colors)
        # Convert from float [0,1] to uint16 [0,65535] as expected by LAS
        las.red = (colors[:, 0] * 65535).astype(np.uint16)
        las.green = (colors[:, 1] * 65535).astype(np.uint16)
        las.blue = (colors[:, 2] * 65535).astype(np.uint16)

    las.write(file_path)
    print(f"Saved point cloud to {file_path} with {len(points)} points (colors saved: {save_colors and pcd.has_colors()}).")

def load_target_with_normals(target_path, downsample_voxel_size=None):  
    if downsample_voxel_size is not None:
        temp_las = target_path.replace('.laz', '_temp_downsampled.laz')
        las = laspy.read(target_path)
        points = np.vstack((las.x, las.y, las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        # save the downsampled point cloud back to a temporary LAS for PDAL processing
        save_o3d_to_las(pcd, temp_las)

        target_path = temp_las

    # check if the temp file already exists, if not, compute normals   
    temp_file = target_path.replace('.laz', '_temp_with_normals.bpf')
    print("Computing normals with PDAL...")
    # PDAL pipeline to compute normals
    pipeline_json = [
        {
            "type": "readers.las",
            "filename": target_path
        },
        {
            "type": "filters.normal",
            "knn": 8
        },
        {
            "type": "writers.bpf",
            "filename": temp_file,
            "output_dims": "X,Y,Z,NormalX,NormalY,NormalZ,Curvature"
        }
    ]
    
    # Execute PDAL pipeline
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()
    
    # Load the file with computed normals
    # Since we used BPF format, we need to read it back with PDAL
    bpf_pipeline = [
        {
            "type": "readers.bpf",
            "filename": temp_file
        }
    ]
    bpf_reader = pdal.Pipeline(json.dumps(bpf_pipeline))
    bpf_reader.execute()
    arrays = bpf_reader.arrays[0]
    
    # Extract points and normals from the PDAL array
    points = np.vstack((arrays['X'], arrays['Y'], arrays['Z'])).T
    normals = np.vstack((arrays['NormalX'], arrays['NormalY'], arrays['NormalZ'])).T

    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    
    print("Normals computed with PDAL and loaded into Open3D point cloud")
    
    return pcd

def align_point_clouds(source_path, threshold, target, voxel_size):
    # Load point clouds
    source = load_las_to_o3d(source_path, downsample_voxel_size=voxel_size)
    
    # Initial transformation (identity matrix)
    init_transform = np.eye(4)

    # Robust ICP parameters
    max_correspondence_distance = threshold
    loss = o3d.pipelines.registration.TukeyLoss(k=voxel_size * 3)
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)

    # Perform robust ICP
    print("Starting robust ICP alignment...")
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, init_transform,
        estimation_method, criteria
    )

    # Print transformation matrix
    print("Transformation Matrix:")
    print(result.transformation)

    return result.transformation

data_folder_path = "../data"
data_folder = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19", "R20", "R21"]
data_files = [os.path.join(data_folder_path, folder, f"{folder}.laz") for folder in data_folder]

def robust_icp():
    pc_files = [os.path.join(data_folder_path, folder, f"{folder}_downsampled_SOR_cropped_centered.laz") for folder in data_folder]
    reference_pc_file = os.path.join(data_folder_path, "R22", "R22_downsampled_SOR_cropped_centered.laz")

    voxel_size = 0.2  # Adjust voxel_size as needed
    target = load_target_with_normals(reference_pc_file, downsample_voxel_size=voxel_size)

    for pc_file in pc_files[:1]:
        print(f"Aligning {pc_file} to {reference_pc_file}...")
        transformation = align_point_clouds(pc_file, threshold=0.5, target=target, voxel_size=voxel_size)
        save_transformation_path = pc_file.replace(".laz", "_transformation.npy")
        np.save(save_transformation_path, transformation)


def icp():
    pc_files = [os.path.join(data_folder_path, folder, f"{folder}_downsampled_SOR_cropped_centered.laz") for folder in data_folder]
    reference_pc_file = os.path.join(data_folder_path, "R22", "R22_downsampled_SOR_cropped_centered.laz")

    voxel_size = 0.2  # Adjust voxel_size as needed
    target = load_las_to_o3d(reference_pc_file, downsample_voxel_size=voxel_size)
    # estimate normals for the target using open3d
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    
    for pc_file in pc_files[:1]:
        source = load_las_to_o3d(pc_file, downsample_voxel_size=voxel_size)
        print(f"Aligning {pc_file} to {reference_pc_file}...")
        threshold = 0.5
        transformation = np.eye(4)
        
        # point to plane ICP
        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        print("Transformation Matrix:")
        print(result.transformation)
    

def apply_transformation():
    pc_files = [os.path.join(data_folder_path, folder, f"{folder}_downsampled_SOR_cropped_centered.laz") for folder in data_folder]

    for pc_file in pc_files[:1]:
        transformation_path = pc_file.replace(".laz", "_transformation.npy")
        if not os.path.exists(transformation_path):
            print(f"Transformation file {transformation_path} not found. Skipping.")
            continue

        transformation = np.load(transformation_path)
        pc = load_las_to_o3d(pc_file, load_colors=True)
        pc.transform(transformation)
        save_o3d_to_las(pc, pc_file.replace(".laz", "_transformed.laz"), save_colors=True)

if __name__ == "__main__":
    #robust_icp()
    #apply_transformation()
    icp()
