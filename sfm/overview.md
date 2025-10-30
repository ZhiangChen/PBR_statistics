# SfM Overview
This project focuses on evaluating the quality of Structure-from-Motion (SfM) point clouds from various datasets, with metrics such as point density and completeness. The evaluation process begins with data preprocessing to clean and standardize the point clouds by removing noise, outliers, and irrelevant points, ensuring reliable analysis. Robust Iterative Closest Point (ICP) is then applied to align the point clouds into the same coordinate system. Following alignment, the area of interest may be refined. Finally, the quality of the SfM point clouds is assessed based on the defined metrics, providing insights into the reconstruction quality.



## 1. Agisoft Metashape

See [SfM Dataset Candidates](./sfm_dataset_candidates.md) for details on how the experiment groups are designed.

See [Metashape Configuration](./metashape_configuration.md) for configuration details.

Point clouds should be exported with WGS 84 / UTM zone 11N (EPSG::32611)

## 2. Robust ICP

2.1 Preprocessing
- Downsample the point cloud
- Remove outliers
- Find the center of reference point cloud
- Centralize the point clouds using the reference center
- Define area of interest
- Crop AoI

2.2 Robust ICP
- Robust ICP
- Refine AoI

## 3. Evaluation
