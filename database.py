from utils import *
from filtering import *
import pandas as pd
from pyproj import Proj, transform
from tqdm import tqdm

def build_database(las_file_path, csv_file_path):
    """
    Build a database from the given LAS file and save it to a CSV file.
    """
    points_semantics_source, colors = read_las_file(las_file_path)
    points_semantics = points_semantics_source.copy()
    results = calculate_hwr(points_semantics_source, n_jobs=8)
    semantics_hwr = {}
    for result in results:
        semantics_id = int(result[0])
        hwr_value = float(result[1])
        semantics_hwr[semantics_id] = hwr_value

    id_to_xyz = group_xyz_by_semantics(points_semantics)
    proj_from = Proj(init='epsg:32611')
    proj_to = Proj(init='epsg:4326')    

    rows = []
    for semantics_id, hwr_value in tqdm(semantics_hwr.items(), desc="Building database"):
        xyz = id_to_xyz[semantics_id]
        x, y = transform(proj_from, proj_to, xyz[:, 0], xyz[:, 1])
        z = xyz[:, 2]

        rows.append({
            "latitude": y.mean(),
            "longitude": x.mean(),
            "altitude": z.mean(),
            "semantics_id": semantics_id,
            "hwr_value": hwr_value
        })

    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(csv_file_path, index=False)
    print(f"Database saved to {csv_file_path}")



if __name__ == "__main__":
    las_file_path = "data/hwr_map/mission_a_density_sor_annotation_hwr_sorted.las"
    csv_file_path = "data/hwr_map/hwr_database.csv"
    build_database(las_file_path, csv_file_path)