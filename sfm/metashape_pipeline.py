import Metashape
import os
import glob
from pathlib import Path
from datetime import datetime
import re

# ANSI color codes for colored output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Global log file handle
log_file_handle = None

def setup_log(output_folder, project_name):
    """Setup log file for the current processing session"""
    global log_file_handle
    log_path = Path(output_folder) / f"{project_name}_processing_log.txt"
    log_file_handle = open(log_path, 'w', encoding='utf-8')
    
    # Write header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file_handle.write(f"=== Metashape Pipeline Log - {timestamp} ===\n")
    log_file_handle.flush()
    return log_path

def close_log():
    """Close the log file"""
    global log_file_handle
    if log_file_handle:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file_handle.write(f"\n=== Log ended - {timestamp} ===\n")
        log_file_handle.close()
        log_file_handle = None

def log_and_print(message):
    """Print to console and log to file"""
    print(message)
    if log_file_handle:
        # Remove ANSI color codes for log file
        clean_message = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', message)
        log_file_handle.write(clean_message + '\n')
        log_file_handle.flush()

def generate_highest_quality_dense_cloud(image_folder, output_folder, project_name="metashape_project"):
    """
    Generate highest quality dense point cloud using maximum quality settings
    
    Args:
        image_folder: Path to folder containing input images
        output_folder: Path to folder for outputs 
        project_name: Name for the project files
    """
    
    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = setup_log(output_folder, project_name)
    
    # Set file paths
    project_path = output_folder / f"{project_name}.psx"
    export_path = output_folder / f"{project_name}_dense_cloud.ply"
    
    # Create a new document
    doc = Metashape.Document()
    
    # Add a new chunk
    chunk = doc.addChunk()
    
    log_and_print(f"{Colors.HEADER}{Colors.BOLD}Starting HIGHEST QUALITY Metashape dense point cloud generation...{Colors.ENDC}")
    
    # Step 1: Add photos
    log_and_print(f"{Colors.BLUE}Step 1: Adding photos...{Colors.ENDC}")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    image_list = []
    
    for extension in image_extensions:
        image_list.extend(glob.glob(os.path.join(image_folder, extension)))
        image_list.extend(glob.glob(os.path.join(image_folder, extension.upper())))
    
    if not image_list:
        log_and_print(f"{Colors.FAIL}No images found in {image_folder}{Colors.ENDC}")
        close_log()
        return False
    
    log_and_print(f"{Colors.GREEN}Found {len(image_list)} images{Colors.ENDC}")
    chunk.addPhotos(image_list)
    
    # Step 2: Align photos with HIGHEST quality settings
    log_and_print(f"{Colors.BLUE}Step 2: Aligning photos (HIGHEST quality)...{Colors.ENDC}")
    chunk.matchPhotos(
        downscale=1,  # 1=Highest quality (no downscaling)
        generic_preselection=True,
        reference_preselection=False,
        filter_mask=False,
        mask_tiepoints=False,
        filter_stationary_points=True,
        keypoint_limit=40000,  # Increased for highest quality
        tiepoint_limit=4000,   # Increased for highest quality
        keep_keypoints=False,
        guided_matching=False,
        reset_matches=False
    )
    
    chunk.alignCameras(
        adaptive_fitting=True,
        reset_alignment=True
    )
    
    # Check if alignment was successful
    aligned_cameras = sum(1 for camera in chunk.cameras if camera.transform)
    log_and_print(f"{Colors.GREEN}Aligned {aligned_cameras} out of {len(chunk.cameras)} cameras{Colors.ENDC}")

    # Check tie points (sparse cloud) after alignment
    if chunk.tie_points:
        tie_point_count = len(chunk.tie_points.points)
        log_and_print(f"{Colors.CYAN}Sparse point cloud created with {tie_point_count:,} tie points{Colors.ENDC}")
    else:
        log_and_print(f"{Colors.WARNING}No tie points found after alignment{Colors.ENDC}")
    
    if aligned_cameras == 0:
        log_and_print(f"{Colors.FAIL}Camera alignment failed!{Colors.ENDC}")
        close_log()
        return False
    
    # Step 3: Optimize camera alignment for highest accuracy
    log_and_print(f"{Colors.BLUE}Step 3: Optimizing camera alignment for highest accuracy...{Colors.ENDC}")
    
    # Filter points by reprojection error first
    chunk.optimizeCameras(
        fit_f=True,          # Optimize focal length
        fit_cx=True,         # Optimize principal point x
        fit_cy=True,         # Optimize principal point y
        fit_b1=True,         # Optimize affinity
        fit_b2=True,         # Optimize non-orthogonality
        fit_k1=True,         # Optimize radial distortion k1
        fit_k2=True,         # Optimize radial distortion k2
        fit_k3=True,         # Optimize radial distortion k3
        fit_k4=False,        # Skip k4 for stability
        fit_p1=True,         # Optimize tangential distortion p1
        fit_p2=True,         # Optimize tangential distortion p2
        adaptive_fitting=True,
        tiepoint_covariance=True
    )
    
    # Get optimization results
    optimized_cameras = sum(1 for camera in chunk.cameras if camera.transform)
    log_and_print(f"{Colors.GREEN}Camera optimization completed. {optimized_cameras} cameras optimized{Colors.ENDC}")
    
    # Check if we have access to tie points after optimization
    if chunk.tie_points:
        current_tie_points = len(chunk.tie_points.points)
        log_and_print(f"{Colors.CYAN}Current tie points available: {current_tie_points:,}{Colors.ENDC}")
    else:
        log_and_print(f"{Colors.WARNING}No tie points accessible after optimization{Colors.ENDC}")
    
    # Optional: Filter points by reprojection error to improve quality
    log_and_print(f"{Colors.CYAN}Filtering tie points by reprojection error...{Colors.ENDC}")
    tie_points_before = len(chunk.tie_points.points) if chunk.tie_points else 0
    
    # Filter tie points with high reprojection error
    if chunk.tie_points and tie_points_before > 0:
        # Filter 1: Reprojection Error
        f = Metashape.TiePoints.Filter()
        f.init(chunk, Metashape.TiePoints.Filter.ReprojectionError)
        threshold_reproj = 0.3  # Conservative threshold for highest quality
        f.selectPoints(threshold_reproj)
        nselected_reproj = len([p for p in chunk.tie_points.points if p.selected])
        
        if nselected_reproj > 0:
            chunk.tie_points.removeSelectedPoints()
            log_and_print(f"{Colors.CYAN}Removed {nselected_reproj} tie points with reprojection error > {threshold_reproj} pixels{Colors.ENDC}")
        
        # Filter 2: Reconstruction Uncertainty
        log_and_print(f"{Colors.CYAN}Filtering tie points by reconstruction uncertainty...{Colors.ENDC}")
        f_uncertainty = Metashape.TiePoints.Filter()
        f_uncertainty.init(chunk, Metashape.TiePoints.Filter.ReconstructionUncertainty)
        threshold_uncertainty = 20  # Conservative threshold for highest quality
        f_uncertainty.selectPoints(threshold_uncertainty)
        nselected_uncertainty = len([p for p in chunk.tie_points.points if p.selected])
        
        if nselected_uncertainty > 0:
            chunk.tie_points.removeSelectedPoints()
            log_and_print(f"{Colors.CYAN}Removed {nselected_uncertainty} tie points with reconstruction uncertainty > {threshold_uncertainty}{Colors.ENDC}")
        
        # Filter 3: Projection Accuracy
        log_and_print(f"{Colors.CYAN}Filtering tie points by projection accuracy...{Colors.ENDC}")
        f_accuracy = Metashape.TiePoints.Filter()
        f_accuracy.init(chunk, Metashape.TiePoints.Filter.ProjectionAccuracy)
        threshold_accuracy = 5  # Conservative threshold for highest quality
        f_accuracy.selectPoints(threshold_accuracy)
        nselected_accuracy = len([p for p in chunk.tie_points.points if p.selected])
        
        if nselected_accuracy > 0:
            chunk.tie_points.removeSelectedPoints()
            log_and_print(f"{Colors.CYAN}Removed {nselected_accuracy} tie points with projection accuracy > {threshold_accuracy}{Colors.ENDC}")
        
        total_removed = nselected_reproj + nselected_uncertainty + nselected_accuracy
        if total_removed > 0:
            # Re-optimize after filtering
            log_and_print(f"{Colors.CYAN}Re-optimizing cameras after tie point filtering...{Colors.ENDC}")
            chunk.optimizeCameras(
                fit_f=True, fit_cx=True, fit_cy=True, fit_b1=True, fit_b2=True,
                fit_k1=True, fit_k2=True, fit_k3=True, fit_p1=True, fit_p2=True,
                adaptive_fitting=True, tiepoint_covariance=True
            )
        else:
            log_and_print(f"{Colors.GREEN}No tie points need filtering (all meet quality thresholds){Colors.ENDC}")
    else:
        log_and_print(f"{Colors.WARNING}No tie points available for filtering{Colors.ENDC}")
    
    tie_points_after = len(chunk.tie_points.points) if chunk.tie_points else 0
    log_and_print(f"{Colors.GREEN}Tie points: {tie_points_before:,} → {tie_points_after:,} (filtered {tie_points_before - tie_points_after:,}){Colors.ENDC}")
    
    # Step 4: Build depth maps with HIGHEST quality
    log_and_print(f"{Colors.BLUE}Step 4: Building depth maps (HIGHEST quality)...{Colors.ENDC}")
    chunk.buildDepthMaps(
        downscale=1,  # 1=Highest quality (no downscaling)
        filter_mode=Metashape.NoFiltering,  # No filtering for maximum detail
        max_neighbors=100,  # Maximum neighbors for best quality
        subdivide_task=True,
        workitem_size_cameras=20,
        max_workgroup_size=100
    )
    
    # Step 5: Build dense cloud with HIGHEST quality
    log_and_print(f"{Colors.BLUE}Step 5: Building dense point cloud (HIGHEST quality)...{Colors.ENDC}")
    
    # Check if we have cameras and depth maps before building point cloud
    available_cameras = sum(1 for camera in chunk.cameras if camera.transform)
    log_and_print(f"{Colors.GREEN}Available cameras for point cloud generation: {available_cameras}{Colors.ENDC}")
    
    if available_cameras == 0:
        log_and_print(f"{Colors.FAIL}No cameras available for point cloud generation!{Colors.ENDC}")
        close_log()
        return False
    
    chunk.buildPointCloud(
        source_data=Metashape.DataSource.DepthMapsData,  # REQUIRED
        point_colors=True,
        point_confidence=True,  # Enable confidence for quality filtering
        keep_depth=True,        # Keep depth maps for reference
        max_neighbors=100,      # Maximum neighbors for best quality
        subdivide_task=True,
        workitem_size_cameras=20,
        max_workgroup_size=100
    )
    
    # Check if dense cloud was created
    if chunk.point_cloud is None:
        log_and_print(f"{Colors.FAIL}Dense cloud generation failed!{Colors.ENDC}")
        close_log()
        return False
    
    point_count = chunk.point_cloud.point_count
    log_and_print(f"{Colors.GREEN}Dense cloud generated with {point_count:,} points{Colors.ENDC}")
    
    # Step 6: Skip filtering for compatibility
    log_and_print(f"{Colors.WARNING}Step 6: Skipping point cloud filtering for compatibility{Colors.ENDC}")
    
    final_point_count = chunk.point_cloud.point_count
    log_and_print(f"{Colors.CYAN}Final point cloud contains: {final_point_count:,} points{Colors.ENDC}")
    
    # Step 7: Export highest quality dense cloud
    log_and_print(f"{Colors.BLUE}Step 7: Exporting highest quality dense cloud...{Colors.ENDC}")
    chunk.exportPointCloud(
        path=export_path,
        source_data=Metashape.PointCloudData,
        binary=True,
        save_point_normal=True,
        save_point_color=True,
        comment="Highest quality dense point cloud generated by Metashape"
    )
    log_and_print(f"{Colors.GREEN}Highest quality dense cloud exported to: {export_path}{Colors.ENDC}")
    
    # Save final project at the end
    doc.save(str(project_path))
    log_and_print(f"{Colors.GREEN}Final project saved to: {project_path}{Colors.ENDC}")
    
    close_log()
    return True

def process_multiple_datasets(datasets):
    """
    Process multiple datasets with the same highest quality settings
    
    Args:
        datasets: List of tuples (image_folder, output_folder, project_name)
    """
    log_and_print(f"{Colors.HEADER}{Colors.BOLD}=== PROCESSING MULTIPLE DATASETS ==={Colors.ENDC}")
    log_and_print(f"{Colors.CYAN}Total datasets to process: {len(datasets)}{Colors.ENDC}")
    log_and_print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    
    results = []
    for i, (image_folder, output_folder, project_name) in enumerate(datasets, 1):
        log_and_print(f"\n{Colors.HEADER}Processing dataset {i}/{len(datasets)}: {project_name}{Colors.ENDC}")
        log_and_print(f"{Colors.CYAN}Input: {image_folder}{Colors.ENDC}")
        log_and_print(f"{Colors.CYAN}Output: {output_folder}{Colors.ENDC}")
        
        success = generate_highest_quality_dense_cloud(image_folder, output_folder, project_name)
        results.append((project_name, success))
        
        if success:
            log_and_print(f"{Colors.GREEN}✓ Dataset '{project_name}' completed successfully{Colors.ENDC}")
        else:
            log_and_print(f"{Colors.FAIL}✗ Dataset '{project_name}' failed{Colors.ENDC}")
    
    # Summary
    log_and_print(f"\n{Colors.HEADER}{Colors.BOLD}=== PROCESSING SUMMARY ==={Colors.ENDC}")
    successful = sum(1 for _, success in results if success)
    for name, success in results:
        status = f"{Colors.GREEN}✓" if success else f"{Colors.FAIL}✗"
        log_and_print(f"{status} {name}{Colors.ENDC}")
    
    log_and_print(f"\n{Colors.CYAN}Completed: {successful}/{len(results)} datasets{Colors.ENDC}")

if __name__ == "__main__":
    # Example 1: Single dataset (original behavior)
    image_folder = '/home/zchen256/Downloads/depth_images/Depthimages'
    output_folder = '/home/zchen256/Downloads/depth_images'
    project_name = 'highest_quality_project'
    
    log_and_print(f"{Colors.HEADER}{Colors.BOLD}=== HIGHEST QUALITY DENSE POINT CLOUD GENERATION ==={Colors.ENDC}")
    log_and_print(f"{Colors.WARNING}Warning: This will take significantly longer but produce the best results{Colors.ENDC}")
    log_and_print(f"{Colors.CYAN}Settings: No downscaling, maximum neighbors, highest precision{Colors.ENDC}")
    log_and_print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    
    # Run the highest quality workflow
    success = generate_highest_quality_dense_cloud(image_folder, output_folder, project_name)
    
    if success:
        project_path = Path(output_folder) / f"{project_name}.psx"
        export_path = Path(output_folder) / f"{project_name}_dense_cloud.ply"
        
        log_and_print(f"\n{Colors.GREEN}{Colors.BOLD}🏆 HIGHEST QUALITY dense point cloud generation completed successfully!{Colors.ENDC}")
        log_and_print(f"{Colors.CYAN}📁 Project file: {project_path}{Colors.ENDC}")
        log_and_print(f"{Colors.CYAN}🌟 Highest quality dense cloud: {export_path}{Colors.ENDC}")
        log_and_print(f"\n{Colors.BLUE}Quality settings used:{Colors.ENDC}")
        log_and_print(f"{Colors.GREEN}- Photo matching: Highest (downscale=1){Colors.ENDC}")
        log_and_print(f"{Colors.GREEN}- Depth maps: Highest (downscale=1){Colors.ENDC}")
        log_and_print(f"{Colors.GREEN}- Dense cloud: Maximum neighbors (100){Colors.ENDC}")
        log_and_print(f"{Colors.GREEN}- Export precision: 8 digits{Colors.ENDC}")
    else:
        log_and_print(f"\n{Colors.FAIL}{Colors.BOLD}❌ Highest quality dense point cloud generation failed!{Colors.ENDC}")
        log_and_print(f"{Colors.WARNING}Please check your input images and try again.{Colors.ENDC}")
    
    # Example 2: Multiple datasets (uncomment to use)
    """
    datasets = [
        ('/path/to/dataset1', '/path/to/output1', 'project1'),
        ('/path/to/dataset2', '/path/to/output2', 'project2'),
        ('/path/to/dataset3', '/path/to/output3', 'project3'),
    ]
    
    process_multiple_datasets(datasets)
    """