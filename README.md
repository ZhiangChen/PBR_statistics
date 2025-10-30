# PBR_statistics

A Python toolkit for analyzing point cloud data from photogrammetric rock reconstructions, focusing on geometric fragility and statistical analysis.

## File Descriptions

#### `database.py`
Database construction module that processes LAS point cloud files to build structured datasets. Reads semantically annotated 3D point clouds, calculates Height-to-Width Ratio (HWR) values using parallel processing, performs coordinate transformations (UTM to WGS84), and exports results to CSV format with geographic statistics.

#### `utils.py`
Core utility module providing fundamental point cloud processing functions including LAS file I/O, data preprocessing with semantic ID management, bounding box calculations, and efficient parallel processing for large datasets. Handles reading/writing multiple LAS files and spatial sorting operations.

#### `filtering.py`
Advanced point cloud filtering and analysis module containing DBSCAN clustering algorithms, statistical outlier removal (SOR), and the core Height-to-Width Ratio calculation using PCA for rock geometry analysis. Includes 3D visualization tools and parallel processing for efficient computation across large datasets.

#### `crs_conversion.ipynb`
Coordinate Reference System conversion utilities for geographic data processing. Converts DMS (Degrees, Minutes, Seconds) format to decimal degrees and transforms coordinates between different spatial reference systems, specifically handling EPSG:6340 projections for georeferencing point cloud data.

#### `pbr_analysis.ipynb`
Comprehensive analysis notebook for Precariously Balanced Rock (PBR) geometry studies. Processes multiple mission datasets with density filtering, computes HWR distributions across rock populations, creates publication-ready visualizations, and performs PCA-based shape analysis with statistical summaries and comparative analysis between datasets.


## Dependencies
Python ≥3.8 and ≤3.12
```
conda create -n pbr_statistics python=3.10
conda activate pbr_statistics
conda install -c conda-forge open3d opencv laspy pandas tqdm pyproj scikit-learn joblib seaborn matplotlib -y
conda install -c conda-forge shapely fiona pdal -y
```

