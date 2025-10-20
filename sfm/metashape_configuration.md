# Metashape Structure-from-Motion Configuration

## Overview

This document the Metashape batch processing configuration used for high-quality structure-from-motion (SfM) reconstruction. 

## Batch Job Configuration

```xml
<batchjobs version="2.1.1">
  <job name="AlignPhotos" target="all">
    <adaptive_fitting>true</adaptive_fitting>
    <downscale>0</downscale>
    <filter_stationary_points>false</filter_stationary_points>
    <mask_tiepoints>false</mask_tiepoints>
  </job>
  <job name="OptimizeCameras" target="all"/>
  <job name="BuildPointCloud" target="all">
    <downscale>1</downscale>
    <reuse_depth>true</reuse_depth>
    <uniform_sampling>false</uniform_sampling>
  </job>
</batchjobs>
```

## Processing Pipeline Steps

### 1. Photo Alignment (`AlignPhotos`)

This step establishes the geometric relationships between images by identifying matching features and computing camera positions and orientations.

#### Key Parameters:

**`adaptive_fitting: true`**
- **Purpose**: Enables adaptive fitting during bundle adjustment
- **Impact**: Allows the optimization algorithm to adaptively adjust camera parameters based on image content and quality
- **Benefits**: Improves accuracy for datasets with varying image quality or complex scenes
- **Trade-off**: Slightly longer processing time but better geometric accuracy

**`downscale: 0`**
- **Purpose**: Controls image resolution used for feature detection and matching
- **Value 0**: Uses full resolution (no downscaling)
- **Impact**: Maximum detail preservation for feature detection
- **Benefits**: Better feature matching accuracy, especially for small or distant objects
- **Trade-off**: Significantly longer processing time and higher memory usage

**`filter_stationary_points: false`**
- **Purpose**: Controls filtering of stationary tie points during alignment
- **Impact**: Retains all detected tie points, including those that may not move significantly
- **Benefits**: Preserves potentially useful geometric constraints
- **Use case**: Recommended for datasets where all tie points contribute to scene geometry

**`mask_tiepoints: false`**
- **Purpose**: Controls whether tie points are masked using image masks
- **Impact**: Processes tie points across entire image area
- **Benefits**: Maximizes available geometric information
- **Prerequisites**: Assumes clean images without significant masking requirements

### 2. Camera Optimization (`OptimizeCameras`)

This step refines the estimated camera parameters using bundle adjustment to minimize reprojection errors.

#### Default Parameters:
- Uses Metashape's default optimization settings
- Optimizes focal length, principal point, distortion parameters, and camera positions
- Applies iterative refinement until convergence

#### Impact:
- Improves geometric accuracy of the reconstruction
- Reduces systematic errors in camera calibration
- Essential for high-precision applications

### 3. Dense Point Cloud Generation (`BuildPointCloud`)

This step creates a dense 3D point cloud by computing depth information for each pixel and triangulating the results.

#### Key Parameters:

**`downscale: 1`**
- **Purpose**: Controls resolution of depth map computation
- **Value 1**: Uses full resolution (highest quality)
- **Impact**: Maximum detail preservation in the dense point cloud
- **Benefits**: Finest possible geometric resolution
- **Trade-off**: Longest processing time and largest output files

**`reuse_depth: true`**
- **Purpose**: Controls whether to reuse existing depth maps
- **Impact**: Leverages depth information from previous processing
- **Benefits**: Faster processing when depth maps already exist
- **Use case**: Recommended for iterative workflows or when depth maps are pre-computed

**`uniform_sampling: false`**
- **Purpose**: Controls point cloud sampling strategy
- **Impact**: Uses adaptive sampling based on surface complexity
- **Benefits**: Higher point density in detailed areas, lower density in uniform areas
- **Result**: More efficient representation while preserving geometric detail
