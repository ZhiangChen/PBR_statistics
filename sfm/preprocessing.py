import open3d as o3d
import numpy as np
import laspy
import os
import pdal 
import fiona
import json
from shapely.geometry import shape, mapping

data_folder_path = "../data"
data_folder = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19", "R20", "R21", "R22"]

data_files = [os.path.join(data_folder_path, folder, f"{folder}.laz") for folder in data_folder]


def load_las_to_o3d(file_path, fixed_center=None, downsample_voxel_size=None):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T

    if points.size == 0:
        raise ValueError(f"{file_path}: loaded 0 points – is the file empty or unreadable?")


    if fixed_center is not None:
        points -= fixed_center

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Load colors if available
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack((las.red, las.green, las.blue)).T
        # Normalize colors from 0-65535 to 0-1 for Open3D
        colors = colors.astype(np.float32) / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"Loaded {file_path} with {len(points)} points and colors.")
    else:
        print(f"Loaded {file_path} with {len(points)} points (no colors).")

    if downsample_voxel_size is not None:
        before = np.asarray(pcd.points).shape[0]
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        after = np.asarray(pcd.points).shape[0]
        print(f"[Info] Voxel downsample {file_path}: {before} -> {after} points (voxel={downsample_voxel_size}).")
        if after == 0:
            raise ValueError(f"{file_path}: voxel downsample produced 0 points. "
                             f"Try a smaller voxel or use PDAL streaming.")

    return pcd

def save_o3d_to_las(pcd, file_path, fixed_center=None):
    """
    Save an Open3D point cloud to a LAS file with millimeter-level accuracy.

    Parameters:
    - pcd: open3d.geometry.PointCloud
    - file_path: Output LAS file path
    - fixed_center: If provided, it will be added back to restore original coordinates
    """
    points = np.asarray(pcd.points)

    if fixed_center is not None:
        points += fixed_center

    # Create LAS header with high precision
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])  # 1 millimeter resolution
    header.offsets = np.min(points, axis=0)  # minimize coordinate compression error

    # Create and fill LAS data
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Save colors if available
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        # Scale colors from 0-1 to 0-65535 for LAS
        colors = (colors * 65535.0).astype(np.uint16)
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]
        print(f"Saved point cloud to {file_path} with {len(points)} points and colors.")
    else:
        print(f"Saved point cloud to {file_path} with {len(points)} points (no colors).")

    las.write(file_path)



def downsample_las_file(input_las_path, output_las_path, voxel_size):
    """
    Load a LAS file, downsample it using voxel grid filtering, and save the result.

    Parameters:
    - input_las_path: Path to the input LAS file
    - output_las_path: Path to save the downsampled LAS file
    - voxel_size: Voxel size for downsampling
    """
    pcd = load_las_to_o3d(input_las_path, downsample_voxel_size=voxel_size)
    save_o3d_to_las(pcd, output_las_path)


def downsample_las_file_pdal(input_las_path, output_las_path, voxel_size):
    """
    Downsample a LAS file using PDAL's voxel-based filtering for efficient processing.

    Parameters:
    - input_las_path: Path to the input LAS file
    - output_las_path: Path to save the downsampled LAS file
    - voxel_size: Voxel size for downsampling (in the same units as the point cloud)
    """

    # Create PDAL pipeline for voxel-based downsampling
    pipeline_json = [
        {
            "type": "readers.las",
            "filename": input_las_path
        },
        {
            "type": "filters.voxeldownsize",
            "cell": voxel_size
        },
        {
            "type": "writers.las",
            "filename": output_las_path,
            "scale_x": 0.001,
            "scale_y": 0.001,
            "scale_z": 0.001,
            "offset_x": "auto",
            "offset_y": "auto",
            "offset_z": "auto"
        }
    ]

    # Execute the pipeline
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()

    # Get metadata about the processing
    metadata = pipeline.metadata
    
    # Extract point counts with multiple fallback approaches
    input_count = "unknown"
    output_count = "unknown"
    
    # Try different metadata access patterns
    try:
        # Standard PDAL metadata structure
        input_count = metadata['metadata']['readers.las']['count']
        output_count = metadata['metadata']['writers.las']['count']
    except KeyError:
        try:
            # Alternative structure with array indexing
            input_count = metadata['metadata']['readers.las'][0]['count']
            output_count = metadata['metadata']['writers.las'][0]['count']
        except (KeyError, IndexError, TypeError):
            try:
                # Direct access without 'metadata' wrapper
                input_count = metadata['readers']['las']['count']
                output_count = metadata['writers']['las']['count']
            except KeyError:
                # Last resort: use pipeline arrays or read output file
                try:
                    arrays = pipeline.arrays
                    input_count = len(arrays[0]) if arrays else "unknown"
                    
                    # Try to read output file to get count
                    try:
                        output_las = laspy.read(output_las_path)
                        output_count = len(output_las.x)
                    except:
                        output_count = "unknown"
                except:
                    pass
    
    # Print results
    if input_count != "unknown" and output_count != "unknown":
        print(f"Downsampled {input_las_path} from {input_count:,} to {output_count:,} points")
        print(f"Reduction ratio: {input_count/output_count:.1f}:1")
    elif input_count != "unknown":
        print(f"Downsampled {input_las_path} from {input_count:,} points (output count unknown)")
    else:
        print(f"Downsampled {input_las_path} (point counts not available in metadata)")
    
    print(f"Saved downsampled file to {output_las_path}")
    



def crop_las_by_shapefile(input_las_file, shapefile_path, name, output_las_path, outside=False):
    """
    Crop a LAS/LAZ file using a polygon from a shapefile.

    Parameters
    ----------
    input_las_file : str
        Path to input LAS/LAZ file.
    shapefile_path : str
        Path to shapefile (.shp) containing polygons.
    name : str
        Name attribute of the polygon feature to use for cropping.
    output_las_path : str
        Path to output cropped LAS/LAZ file.
    outside : bool, optional
        If True, keep points outside the polygon instead of inside.

    Returns
    -------
    bool
        True if cropping succeeded, False otherwise.
    """
    # --- Step 1: Find target polygon ---
    polygon = None
    with fiona.open(shapefile_path) as shapefile:
        for feature in shapefile:
            if feature["properties"].get("name") == name:
                polygon = mapping(shape(feature["geometry"]))  # GeoJSON-style dict
                break

    if polygon is None:
        print(f"[Error] Polygon with name '{name}' not found in {shapefile_path}")
        return False

    # --- Step 2: Build PDAL pipeline ---
    pipeline_def = {
        "pipeline": [
            {"type": "readers.las", "filename": input_las_file},
            {"type": "filters.crop", "polygon": polygon, "outside": outside},
            {"type": "writers.las", "filename": output_las_path},
        ]
    }

    # --- Step 3: Execute PDAL pipeline ---
    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_def))
        pipeline.execute()
        print(f"[OK] Cropped LAS saved to {output_las_path}")
        return True
    except Exception as e:
        print(f"[Error] PDAL execution failed: {e}")
        return False


def main():
    downsample_flag = False
    SOR_flag = False
    centeralize_flag = True
    crop_flag = True

    voxel_size = 0.05  # Example voxel size for downsampling (in meters)
    center = np.array([-281222.0, -4269777.0, -1757.0])  # Example center for centralization

    if downsample_flag:
        for file_path in data_files:
            folder_name = os.path.basename(os.path.dirname(file_path))
            output_las_path = os.path.join(os.path.dirname(file_path), f"{folder_name}_downsampled.laz")
            #downsample_las_file(file_path, output_las_path, voxel_size)
            # downsample using pdal method
            downsample_las_file_pdal(file_path, output_las_path, voxel_size)
            

    downsampled_files = [os.path.join(data_folder_path, folder, f"{folder}_downsampled.laz") for folder in data_folder]
    for file_path in downsampled_files:
    
        save_file_path = file_path.replace("_downsampled.laz", "_downsampled_SOR.laz")
        if SOR_flag:
            pcd = load_las_to_o3d(file_path)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)
            # Save the filtered point cloud back to LAS
            save_o3d_to_las(pcd, save_file_path)
            file_path = save_file_path
        else:
            file_path = save_file_path
            pcd = load_las_to_o3d(file_path)
        

        cropped_file_path = file_path.replace(".laz", "_cropped.laz")
        if crop_flag:
            crop_las_by_shapefile(
                input_las_file=file_path,
                shapefile_path="../data/aoi/cb.shp",
                name="aoi",
                output_las_path=cropped_file_path,
                outside=False
            )
        file_path = cropped_file_path
        
        save_file_path = file_path.replace(".laz", "_centered.laz")
        if centeralize_flag:
            pcd = load_las_to_o3d(file_path)
            save_o3d_to_las(pcd, save_file_path, fixed_center=center)


if __name__ == "__main__":
    main()
    # input_las = "../data/R12/R12.laz"
    # output_las = "../data/R12/R12_downsampled.laz"
    # downsample_las_file_pdal(input_las, output_las, voxel_size=0.05)
    
    # Uncomment to delete temporary files
    # delete_temp_files()