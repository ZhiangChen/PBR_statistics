#!/usr/bin/env python3
"""
Voxel Density Analysis for 3D Point Clouds

This program analyzes the density distribution of 3D point clouds by:
1. Reading LAZ files (compressed LAS format)
2. Creating a regular voxel grid with specified spacing
3. Counting points in each voxel (including empty voxels)
4. Providing density statistics and visualization
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from collections import defaultdict
import argparse
import os
from pathlib import Path
import time


class VoxelDensityAnalyzer:
    """
    Analyzes point cloud density by voxelization
    """
    
    def __init__(self, voxel_size=1.0, bounds=None):
        """
        Initialize the analyzer
        
        Args:
            voxel_size (float): Size of voxels in the same units as point cloud
            bounds (tuple): Optional bounds (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        self.voxel_size = voxel_size
        self.custom_bounds = bounds  # Store custom bounds if provided
        self.points = None
        self.voxel_counts = None
        self.voxel_grid = None
        self.bounds = None
        self.grid_dims = None
        self.original_las_file = None
        
    def load_laz_file(self, laz_file_path):
        """
        Load point cloud from LAZ file (compressed LAS format) - OPTIMIZED
        
        Args:
            laz_file_path (str): Path to LAZ file
            
        Returns:
            np.ndarray: Point coordinates (N x 3)
        """
        print(f"Loading LAZ file: {laz_file_path}")
        start_time = time.time()
        
        try:
            # Read LAZ file (laspy automatically handles LAZ compression)
            las_file = laspy.read(laz_file_path)
            # Store the original file for later use when saving
            self.original_las_file = las_file
        except Exception as e:
            print(f"Error reading LAZ file: {e}")
            print("Make sure the file is a valid LAZ file and laspy supports LAZ format")
            raise
        
        # Extract coordinates using vectorized operations (most efficient)
        print("Extracting coordinates...")
        self.points = np.column_stack((las_file.x, las_file.y, las_file.z))
        
        load_time = time.time() - start_time
        print(f"Loaded {len(self.points):,} points in {load_time:.2f}s")
        
        # Calculate bounds efficiently in one pass
        print("Calculating bounds...")
        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
        print(f"Point cloud bounds:")
        print(f"  X: {mins[0]:.2f} to {maxs[0]:.2f}")
        print(f"  Y: {mins[1]:.2f} to {maxs[1]:.2f}")
        print(f"  Z: {mins[2]:.2f} to {maxs[2]:.2f}")
        
        # Cache bounds for later use
        self._cached_bounds = (mins, maxs)
        
        return self.points
    
    def create_voxel_grid(self):
        """
        Create voxel grid parameters - OPTIMIZED (no massive array allocation)
        Uses custom bounds if provided, otherwise uses point cloud bounds
        """
        if self.points is None:
            raise ValueError("No points loaded. Call load_laz_file() first.")
        
        if self.custom_bounds is not None:
            # Use custom bounds if provided
            x_min, x_max, y_min, y_max, z_min, z_max = self.custom_bounds
            min_coords = np.array([x_min, y_min, z_min])
            max_coords = np.array([x_max, y_max, z_max])
            print(f"Using custom bounds:")
            print(f"  X: {x_min:.2f} to {x_max:.2f}")
            print(f"  Y: {y_min:.2f} to {y_max:.2f}")
            print(f"  Z: {z_min:.2f} to {z_max:.2f}")
        else:
            # Use pre-calculated bounds from loading step
            if hasattr(self, '_cached_bounds'):
                min_coords, max_coords = self._cached_bounds
            else:
                # Calculate bounding box using vectorized operations
                min_coords = np.min(self.points, axis=0)
                max_coords = np.max(self.points, axis=0)
            
            # Reduce padding to minimize empty voxels
            padding = self.voxel_size * 0.01
            min_coords -= padding
            max_coords += padding
            print(f"Using point cloud bounds with padding:")
            print(f"  X: {min_coords[0]:.2f} to {max_coords[0]:.2f}")
            print(f"  Y: {min_coords[1]:.2f} to {max_coords[1]:.2f}")
            print(f"  Z: {min_coords[2]:.2f} to {max_coords[2]:.2f}")
        
        self.bounds = (min_coords, max_coords)
        
        # Calculate grid dimensions
        grid_size = max_coords - min_coords
        self.grid_dims = np.ceil(grid_size / self.voxel_size).astype(int)
        
        total_voxels = np.prod(self.grid_dims.astype(np.int64))
        print(f"Voxel grid parameters:")
        print(f"  Voxel size: {self.voxel_size}")
        print(f"  Grid dimensions: {self.grid_dims[0]} x {self.grid_dims[1]} x {self.grid_dims[2]}")
        print(f"  Total theoretical voxels: {total_voxels:,}")
        
        return self.grid_dims
    
    def voxelize_points(self):
        """
        OPTIMIZED: Assign points to voxels using vectorized operations
        """
        if self.points is None or self.bounds is None:
            raise ValueError("Grid not created. Call create_voxel_grid() first.")
        
        print("Voxelizing points (vectorized approach)...")
        start_time = time.time()
        
        min_coords, max_coords = self.bounds
        
        # Vectorized coordinate to index conversion
        point_indices = np.floor((self.points - min_coords) / self.voxel_size).astype(np.int32)
        
        # Clip indices to ensure they're within bounds
        point_indices = np.clip(point_indices, 0, self.grid_dims - 1)
        
        # Convert 3D indices to 1D for efficient processing
        print("Converting to linear indices...")
        linear_indices = (point_indices[:, 0].astype(np.int64) * self.grid_dims[1] * self.grid_dims[2] + 
                         point_indices[:, 1].astype(np.int64) * self.grid_dims[2] + 
                         point_indices[:, 2].astype(np.int64))
        
        # Use numpy's unique function for fast counting
        print("Counting unique voxels...")
        unique_indices, counts = np.unique(linear_indices, return_counts=True)
        
        # Convert back to 3D indices for storage (vectorized approach)
        self.voxel_counts = {}
        print("Converting back to 3D coordinates (optimized)...")
        
        # Vectorized conversion back to 3D indices
        k_vals = unique_indices % self.grid_dims[2]
        temp = (unique_indices - k_vals) // self.grid_dims[2]
        j_vals = temp % self.grid_dims[1]
        i_vals = (temp - j_vals) // self.grid_dims[1]
        
        # Build dictionary in one go
        for i, j, k, count in zip(i_vals, j_vals, k_vals, counts):
            self.voxel_counts[(int(i), int(j), int(k))] = int(count)
        
        voxel_time = time.time() - start_time
        print(f"Voxelization completed in {voxel_time:.2f}s")
        print(f"Points distributed across {len(self.voxel_counts):,} non-empty voxels")
        
        # Skip creating the massive 3D array - work with sparse representation only
        total_voxels = np.prod(self.grid_dims.astype(np.int64))
        non_empty_voxels = len(self.voxel_counts)
        empty_voxels = total_voxels - non_empty_voxels
        occupancy_rate = non_empty_voxels / total_voxels * 100
        
        print(f"Voxel statistics:")
        print(f"  Total voxels: {total_voxels:,}")
        print(f"  Non-empty voxels: {non_empty_voxels:,}")
        print(f"  Empty voxels: {empty_voxels:,}")
        print(f"  Occupancy rate: {occupancy_rate:.4f}%")
        
        return self.voxel_counts
    
        # Remove the old slow method calls
        self.create_complete_grid = None  # Not needed anymore
    
    def get_density_statistics(self):
        """
        OPTIMIZED: Calculate density statistics without massive array
        
        Returns:
            dict: Dictionary containing various statistics
        """
        if self.voxel_counts is None:
            raise ValueError("Voxelization not performed. Call voxelize_points() first.")
        
        print("Calculating density statistics (optimized)...")
        start_time = time.time()
        
        # Work directly with voxel counts
        all_counts = np.array(list(self.voxel_counts.values()))
        non_zero_counts = all_counts[all_counts > 0]
        
        # Calculate empty voxels from total grid
        total_voxels = np.prod(self.grid_dims.astype(np.int64))
        empty_voxels = total_voxels - len(all_counts)
        
        # Statistics for non-zero voxels
        if len(non_zero_counts) > 0:
            mean_non_zero = np.mean(non_zero_counts)
            std_non_zero = np.std(non_zero_counts)
            median_non_zero = np.median(non_zero_counts)
            min_density = np.min(non_zero_counts)
            max_density = np.max(non_zero_counts)
            percentile_95 = np.percentile(non_zero_counts, 95)
            percentile_99 = np.percentile(non_zero_counts, 99)
        else:
            mean_non_zero = std_non_zero = median_non_zero = 0
            min_density = max_density = percentile_95 = percentile_99 = 0
        
        # Statistics including all voxels (zeros and non-zeros)
        total_points = np.sum(all_counts)
        mean_all = total_points / total_voxels if total_voxels > 0 else 0
        
        stats = {
            'total_voxels': int(total_voxels),
            'empty_voxels': int(empty_voxels),
            'non_empty_voxels': len(non_zero_counts),
            'occupancy_rate': len(non_zero_counts) / total_voxels * 100 if total_voxels > 0 else 0,
            'mean_density_all': mean_all,
            'mean_density_non_empty': mean_non_zero,
            'std_density_non_empty': std_non_zero,
            'min_density': int(min_density),
            'max_density': int(max_density),
            'median_density_non_empty': median_non_zero,
            'percentile_95_non_empty': percentile_95,
            'percentile_99_non_empty': percentile_99,
            'total_points': int(total_points)
        }
        
        stats_time = time.time() - start_time
        print(f"Statistics calculated in {stats_time:.2f}s")
        
        return stats
    
    def visualize_density_distribution(self, output_path=None):
        """
        OPTIMIZED: Create visualizations without massive arrays
        
        Args:
            output_path (str): Optional path to save plots
        """
        if self.voxel_counts is None:
            raise ValueError("Voxelization not performed. Call voxelize_points() first.")
        
        print("Creating density visualizations (optimized)...")
        start_time = time.time()
        
        # Work with all voxel counts (including zeros if present)
        all_counts = np.array(list(self.voxel_counts.values()))
        non_zero_counts = all_counts[all_counts > 0]
        
        # Use non-zero counts for most visualizations to avoid skewing by zeros
        # But show zero information in titles/labels
        
        # Set up plotting
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Voxel Density Analysis (Voxel Size: {self.voxel_size})', fontsize=16)
        
        # 1. Histogram of non-zero voxel densities
        if len(non_zero_counts) > 0:
            axes[0, 0].hist(non_zero_counts, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Points per Voxel')
        axes[0, 0].set_ylabel('Frequency')
        zero_count = len(all_counts) - len(non_zero_counts)
        axes[0, 0].set_title(f'Density Distribution (Non-zero Voxels)\nZero voxels: {zero_count:,}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Log-scale histogram
        if len(non_zero_counts) > 0:
            axes[0, 1].hist(non_zero_counts, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_xlabel('Points per Voxel')
        axes[0, 1].set_ylabel('Frequency (log scale)')
        axes[0, 1].set_title('Density Distribution (Log Scale)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        if len(non_zero_counts) > 0:
            sorted_counts = np.sort(non_zero_counts)
            axes[1, 0].plot(sorted_counts, np.arange(1, len(sorted_counts) + 1) / len(sorted_counts))
        axes[1, 0].set_xlabel('Points per Voxel')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Box plot (fix matplotlib deprecation warning)
        if len(non_zero_counts) > 0:
            axes[1, 1].boxplot([non_zero_counts], tick_labels=['Non-zero Voxels'])
        axes[1, 1].set_ylabel('Points per Voxel')
        axes[1, 1].set_title('Density Distribution Summary')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Density distribution plot saved to: {output_path}")
        else:
            print("Warning: No output path provided, plot not saved")
        
        viz_time = time.time() - start_time
        print(f"Visualization completed in {viz_time:.2f}s")
        
        # Close the figure to free memory instead of showing
        plt.close(fig)
    
    def export_results(self, output_path):
        """
        Export voxel density results to file
        
        Args:
            output_path (str): Path for output file
        """
        if self.voxel_counts is None:
            raise ValueError("Voxelization not performed. Call voxelize_points() first.")
        
        print(f"Exporting results to: {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get statistics
        stats = self.get_density_statistics()
        min_coords, max_coords = self.bounds
        
        with open(output_path, 'w') as f:
            f.write("Voxel Density Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Parameters:\n")
            f.write(f"  Voxel size: {self.voxel_size}\n")
            f.write(f"  Grid dimensions: {self.grid_dims[0]} x {self.grid_dims[1]} x {self.grid_dims[2]}\n")
            f.write(f"  Bounds: ({min_coords[0]:.2f}, {min_coords[1]:.2f}, {min_coords[2]:.2f}) to ")
            f.write(f"({max_coords[0]:.2f}, {max_coords[1]:.2f}, {max_coords[2]:.2f})\n\n")
            
            f.write("Statistics:\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value:,}\n")
    
    def save_voxelized_point_cloud(self, output_path, subsample_factor=1):
        """
        OPTIMIZED: Save voxels as point cloud where each point is a voxel center
        
        Args:
            output_path (str): Path for output LAZ file  
            subsample_factor (int): Not used for voxel centers (all non-empty voxels saved)
        """
        if self.voxel_counts is None:
            raise ValueError("Voxelization not performed.")
        
        print(f"Saving voxel centers as point cloud to: {output_path}")
        print(f"Each point represents a voxel center with intensity = point count")
        
        start_time = time.time()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert voxel indices to world coordinates (voxel centers) - VECTORIZED
        print("Converting voxel indices to world coordinates (vectorized)...")
        min_coords, max_coords = self.bounds
        
        voxel_indices = list(self.voxel_counts.keys())
        voxel_counts_list = list(self.voxel_counts.values())
        
        # Convert voxel indices to voxel center coordinates using vectorized operations
        voxel_indices_array = np.array(voxel_indices, dtype=np.float64)
        
        # Vectorized calculation: center = min_coords + (index + 0.5) * voxel_size
        voxel_centers = min_coords + (voxel_indices_array + 0.5) * self.voxel_size
        voxel_intensities = np.array(voxel_counts_list, dtype=np.uint32)  # Use uint32 for larger values
        
        print(f"Creating point cloud with {len(voxel_centers):,} voxel centers")
        print(f"Intensity range: {np.min(voxel_intensities)} to {np.max(voxel_intensities)}")
        
        # Create new LAS file for voxel centers
        try:
            # Try with point format 3 (RGB + intensity)
            output_las = laspy.create(point_format=3, file_version="1.2")
        except:
            try:
                # Fallback to simpler creation method
                output_las = laspy.create(point_format=3)
            except:
                # Final fallback to basic point format
                output_las = laspy.create(point_format=2)  # Format 2 has intensity
        
        # Set header information
        output_las.header.point_count = len(voxel_centers)
        output_las.header.min = np.min(voxel_centers, axis=0)
        output_las.header.max = np.max(voxel_centers, axis=0)
        output_las.header.scale = [0.001, 0.001, 0.001]  # 1mm precision
        output_las.header.offset = output_las.header.min
        
        # Set point coordinates
        output_las.x = voxel_centers[:, 0]
        output_las.y = voxel_centers[:, 1] 
        output_las.z = voxel_centers[:, 2]
        
        # Set intensity values to voxel counts
        # Scale down intensity values to fit in uint16 range (0-65535) for LAS format
        max_intensity = np.max(voxel_intensities)
        if max_intensity > 65535:
            # Scale intensities to fit in uint16 range while preserving relative differences
            scale_factor = 65535.0 / max_intensity
            scaled_intensities = (voxel_intensities * scale_factor).astype(np.uint16)
            print(f"Scaling intensities: {max_intensity} -> 65535 (scale factor: {scale_factor:.6f})")
            output_las.intensity = scaled_intensities
        else:
            output_las.intensity = voxel_intensities.astype(np.uint16)
        
        # Set colors based on density (optional - creates a nice visualization)
        print("Setting colors based on density...")
        # Normalize intensities to 0-255 for color mapping
        norm_intensities = (voxel_intensities - np.min(voxel_intensities)) / (np.max(voxel_intensities) - np.min(voxel_intensities))
        
        # Create color map: blue (low density) -> green -> red (high density)
        red = (norm_intensities * 255).astype(np.uint16) * 256  # Scale to 16-bit
        green = ((1 - np.abs(norm_intensities - 0.5) * 2) * 255).astype(np.uint16) * 256
        blue = ((1 - norm_intensities) * 255).astype(np.uint16) * 256
        
        # Only set RGB if point format supports it
        if hasattr(output_las, 'red'):
            output_las.red = red
            output_las.green = green  
            output_las.blue = blue
        else:
            print("Point format doesn't support RGB colors, skipping color assignment")
        
        # Set classification to distinguish voxel centers (class 18 = high vegetation)
        if hasattr(output_las, 'classification'):
            output_las.classification = np.full(len(voxel_centers), 18, dtype=np.uint8)
        
        # Write the file
        print("Writing LAZ file...")
        output_las.write(output_path)
        
        save_time = time.time() - start_time
        print(f"Voxel centers saved in \033[94m{save_time:.2f}s\033[0m!")
        print(f"Saved {len(voxel_centers):,} voxel centers")
        print(f"Intensity range: {np.min(voxel_intensities)} to {np.max(voxel_intensities)} points per voxel")
        print(f"Each point represents a {self.voxel_size} x {self.voxel_size} x {self.voxel_size} voxel")
        
        return output_path
    
    def analyze(self, laz_file_path, output_dir=None):
        """
        Complete analysis workflow
        
        Args:
            laz_file_path (str): Path to LAZ file
            output_dir (str): Optional output directory for results
            
        Returns:
            dict: Analysis statistics
        """
        print(f"Starting voxel density analysis...")
        print(f"Voxel size: {self.voxel_size}")
        total_start_time = time.time()
        
        # Load points
        step_start = time.time()
        self.load_laz_file(laz_file_path)
        load_time = time.time() - step_start
        print(f"⏱️  Step 1 (Load LAZ) completed in \033[94m{load_time:.2f}s\033[0m")
        
        # Create voxel grid
        step_start = time.time()
        self.create_voxel_grid()
        grid_time = time.time() - step_start
        print(f"⏱️  Step 2 (Create Grid) completed in \033[94m{grid_time:.2f}s\033[0m")
        
        # Voxelize points
        step_start = time.time()
        self.voxelize_points()
        voxel_time = time.time() - step_start
        print(f"⏱️  Step 3 (Voxelize) completed in \033[94m{voxel_time:.2f}s\033[0m")
        
        # Get statistics
        step_start = time.time()
        stats = self.get_density_statistics()
        stats_time = time.time() - step_start
        print(f"⏱️  Step 4 (Statistics) completed in \033[94m{stats_time:.2f}s\033[0m")
        
        # Print statistics
        print("\nDensity Statistics:")
        print("-" * 30)
        print(f"📊 VOXEL OCCUPANCY:")
        print(f"   Total voxels: {stats['total_voxels']:,}")
        print(f"   Occupied voxels: {stats['non_empty_voxels']:,}")
        print(f"   Empty voxels: {stats['empty_voxels']:,}")
        print(f"   Occupancy rate: {stats['occupancy_rate']:.4f}%")
        print(f"\n📈 DENSITY STATISTICS:")
        for key, value in stats.items():
            if key not in ['total_voxels', 'empty_voxels', 'non_empty_voxels', 'occupancy_rate']:
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value:,}")
        
        # Generate outputs if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export statistics
            step_start = time.time()
            stats_file = output_dir / f"voxel_density_stats_size_{self.voxel_size}.txt"
            self.export_results(stats_file)
            export_time = time.time() - step_start
            print(f"⏱️  Step 5 (Export Stats) completed in \033[94m{export_time:.2f}s\033[0m")
            
            # Create visualization
            step_start = time.time()
            plot_file = output_dir / f"voxel_density_plot_size_{self.voxel_size}.png"
            self.visualize_density_distribution(plot_file)
            viz_time = time.time() - step_start
            print(f"⏱️  Step 6 (Visualization) completed in \033[94m{viz_time:.2f}s\033[0m")
            
            # Save voxelized point cloud
            step_start = time.time()
            voxelized_file = output_dir / f"voxel_centers_size_{self.voxel_size}.laz"
            self.save_voxelized_point_cloud(voxelized_file)
            save_time = time.time() - step_start
            print(f"⏱️  Step 7 (Save Point Cloud) completed in \033[94m{save_time:.2f}s\033[0m")
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 TOTAL ANALYSIS TIME: \033[94m{total_time:.2f}s\033[0m (\033[94m{total_time/60:.1f} minutes\033[0m)")
        print(f"Processing rate: \033[94m{len(self.points)/total_time:.0f} points/second\033[0m")
        
        return stats


def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(description='Analyze voxel density of point clouds')
    parser.add_argument('laz_file', help='Path to LAZ file')
    parser.add_argument('--voxel_size', type=float, default=1.0, 
                       help='Voxel size (default: 1.0)')
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory for results and plots')
    parser.add_argument('--bounds', type=float, nargs=6, metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX', 'Z_MIN', 'Z_MAX'),
                       help='Custom bounds: x_min x_max y_min y_max z_min z_max')
    
    args = parser.parse_args()
    
    # Create analyzer with bounds if provided
    bounds = tuple(args.bounds) if args.bounds else None
    analyzer = VoxelDensityAnalyzer(voxel_size=args.voxel_size, bounds=bounds)
    
    # Run analysis
    try:
        stats = analyzer.analyze(args.laz_file, args.output_dir)
        print(f"\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Configuration variables - modify these paths as needed
    LAZ_FILE_PATH = "/atomic-data/zhiang/centennial_bluff/R01_GUI_/sampling.laz"  # Change this to your LAZ file path
    OUTPUT_DIR = "/atomic-data/zhiang/centennial_bluff/R01_GUI_"     # Change this to your desired output directory
    VOXEL_SIZE = 0.5                              # Change this to your desired voxel size
    # Optional custom bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
    # Set to None to use point cloud bounds automatically
    CUSTOM_BOUNDS = None  # Example: (100.0, 200.0, 50.0, 150.0, 0.0, 50.0)

    CUSTOM_BOUNDS = (-242317, -242047, 4301179, 4301374, 1661, 1805)
    
    # Create analyzer with specified voxel size and optional bounds
    analyzer = VoxelDensityAnalyzer(voxel_size=VOXEL_SIZE, bounds=CUSTOM_BOUNDS)
    
    # Run analysis
    try:
        print(f"Analyzing LAZ file: {LAZ_FILE_PATH}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Voxel size: {VOXEL_SIZE}")
        if CUSTOM_BOUNDS:
            print(f"Custom bounds: {CUSTOM_BOUNDS}")
        else:
            print("Using automatic point cloud bounds")
        print("-" * 50)
        
        stats = analyzer.analyze(LAZ_FILE_PATH, OUTPUT_DIR)
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {OUTPUT_DIR}")
        
    except FileNotFoundError:
        print(f"Error: LAZ file not found at {LAZ_FILE_PATH}")
        print("Please update the LAZ_FILE_PATH variable with the correct path to your LAZ file.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your file path and ensure the LAZ file is valid.")