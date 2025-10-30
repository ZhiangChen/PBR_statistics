from flight_analysis import * 
from filtering import *
import os
import pandas as pd
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def extract_points(points_semantics, semantics_id, boarder=0.1):
    """
    Extract points with a specific semantics_id and apply a border around them.
    """
    # filter points by semantics_id
    mask = points_semantics[:, 3] == semantics_id
    points_filtered = points_semantics[mask, :3]
    
    # apply border
    if len(points_filtered) > 0:
        if boarder <= 0:
            return points_filtered, mask, None
        else:
            min_coords = np.min(points_filtered, axis=0) - boarder
            max_coords = np.max(points_filtered, axis=0) + boarder
            mask_border = np.all((points_semantics[:, :3] >= min_coords) & (points_semantics[:, :3] <= max_coords), axis=1)
            return points_filtered, mask, mask_border
    else:
        return None, None, None



class RockAnnotationApp:
    def __init__(self, las_file_path):
        self.las_file_path = las_file_path
        self.csv_file_path = f"data/pc_objects/{os.path.basename(las_file_path).replace('.las', '.csv')}"
        self.df = self.load_or_create_csv()
        self.points_semantics, self.colors = read_las_file(self.las_file_path)
        self.results = sorted(calculate_hwr(self.points_semantics, n_jobs=8), key=lambda x: x[1], reverse=True)
        self.index = 0

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Rock Annotation", 1024, 768)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)
        self.panel = gui.Vert(0.35 * self.window.content_rect.width)  
        self.progress_label = gui.Label("Progress: 0 / {}".format(len(self.results)))
        self.panel.add_child(self.progress_label)
        self.label_buttons = gui.Vert()
        self.label_buttons.add_child(gui.Widget())  # flexible spacer pushes buttons to the bottom
        annotation_labels = ['free_standing_rock', 'no_free_standing_rock', 'uncertain']
        # self.label_buttons.add_child(gui.Label(" "))  # Adds vertical spacer above buttons
        for i, label_text in enumerate(["free-standing rock (q)", "no fs rock (w)", "uncertain (e)"]):
            button = gui.Button(label_text.capitalize())
            annotation_label = annotation_labels[i]
            button.set_on_clicked(lambda text=annotation_label: self.on_label_click(text))
            self.label_buttons.add_child(button)
        
        back_button = gui.Button("Back")
        back_button.set_on_clicked(self.on_back_click)
        self.label_buttons.add_child(back_button)
        # Add text edit for notes
        self.note_edit = gui.TextEdit()
        self.note_edit.placeholder_text = "Enter optional note here"
        self.label_buttons.add_child(self.note_edit)
        self.panel.add_child(self.label_buttons)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_key(self.on_key_event)
        self.load_current()

    def on_layout(self, layout_context):
        content_rect = self.window.content_rect
        panel_width = 300
        self.scene.frame = gui.Rect(content_rect.x, content_rect.y,
                                    content_rect.width - panel_width, content_rect.height)
        self.panel.frame = gui.Rect(content_rect.get_right() - panel_width, content_rect.y,
                                    panel_width, content_rect.height)

    def load_or_create_csv(self):
        if os.path.exists(self.csv_file_path):
            return pd.read_csv(self.csv_file_path)
        else:
            df = pd.DataFrame({
                "semantics_id": pd.Series(dtype=int),
                "label": pd.Series(dtype=str),
                "note": pd.Series(dtype=str)
            })
            df.to_csv(self.csv_file_path, index=False)
            return df

    def load_current(self):
        while self.index < len(self.results):
            semantics_id, _ = self.results[self.index]
            if semantics_id in self.df['semantics_id'].values:
                self.index += 1
                continue
            filtered_points, mask, mask_border = extract_points(self.points_semantics, semantics_id, boarder=1.0)
            if filtered_points is None:
                self.df = pd.concat([self.df, pd.DataFrame([{"semantics_id": semantics_id, "label": None}])], ignore_index=True)
                self.df.to_csv(self.csv_file_path, index=False)
                self.index += 1
            else:
                self.progress_label.text = "Progress: {} / {}    \nHWR: {:.3f}\nID: {}".format(
                    self.index + 1, len(self.results), self.results[self.index][1], int(self.results[self.index][0])
                )
                self.display(filtered_points, mask, mask_border, semantics_id)
                return
        gui.Application.instance.quit()

    def display(self, filtered_points, mask, mask_border, semantics_id):
        mask_in_border = mask_border
        points_border = self.points_semantics[mask_in_border, :3]
        colors_border = self.colors[mask_in_border, :3]
        points_border -= np.mean(points_border, axis=0)
        colors_border = (colors_border / 65535.0).astype(np.float64)
        highlight_color = np.array([1.0, 0.0, 0.0])
        blend_ratio = 0.5
        mask_filtered_in_border = mask[mask_border]
        colors_border[mask_filtered_in_border] = (
            (1 - blend_ratio) * colors_border[mask_filtered_in_border] +
            blend_ratio * highlight_color
        )

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points_border)
        self.pcd.colors = o3d.utility.Vector3dVector(colors_border)

        self.scene.scene.clear_geometry()
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.scene.scene.add_geometry("axis", axis, rendering.MaterialRecord())
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 8.0
        self.scene.scene.add_geometry("pcd", self.pcd, mat)
        bounds = self.pcd.get_axis_aligned_bounding_box()
        center = np.mean(points_border, axis=0)
        self.scene.setup_camera(120, bounds, center)
        eye = center + np.array([-1.0, 0.0, 0.0]) * 3  # from -X direction
        up = np.array([0.0, 0.0, 1.0])  # Z is up
        self.scene.scene.camera.look_at(center, eye, up)


    def on_label_click(self, label):
        semantics_id, _ = self.results[self.index]
        note = self.note_edit.text_value
        self.df = pd.concat([self.df, pd.DataFrame([{
            "semantics_id": semantics_id,
            "label": label,
            "note": note
        }])], ignore_index=True)
        self.df.to_csv(self.csv_file_path, index=False)
        self.note_edit.text_value = ""
        self.index += 1
        self.load_current()

    def on_key_event(self, event):
        if event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.KeyName.Q:
                self.on_label_click("free_standing_rock")
                return True
            elif event.key == gui.KeyName.W:
                self.on_label_click("no_free_standing_rock")
                return True
            elif event.key == gui.KeyName.E:
                self.on_label_click("uncertain")
                return True
        return False
    
    def on_back_click(self):
        if self.index > 0:
            self.index -= 1
            self.df = self.df[self.df['semantics_id'] != self.results[self.index][0]]
            self.df.to_csv(self.csv_file_path, index=False)
            self.load_current()

def annotate_rock(las_file_path):
    # check if the file exists
    if not os.path.exists(las_file_path):
        raise FileNotFoundError(f"The file {las_file_path} does not exist.")
    else:
        app = RockAnnotationApp(las_file_path)
        gui.Application.instance.run()

def filter_las_with_annotations(las_file_path, csv_file_path):
    """
    Filter the LAS file based on annotations in the CSV file.
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"The CSV file {csv_file_path} does not exist.")
    
    if not os.path.exists(las_file_path):
        raise FileNotFoundError(f"The LAS file {las_file_path} does not exist.")
    
    df = pd.read_csv(csv_file_path)
    points_semantics, colors = read_las_file(las_file_path)

    unique_semantics, counts = np.unique(points_semantics[:, 3], return_counts=True)
    background_semantics = unique_semantics[np.argmax(counts)]
    
    # Filter points based on annotations: set semantics_id to -1 for no_free_standing_rock annotations
    for _, row in df.iterrows():
        semantics_id = row['semantics_id']
        if row['label'] == 'no_free_standing_rock':
            points_semantics[points_semantics[:, 3] == semantics_id, 3] = background_semantics

    # Save the filtered points back to a new LAS file
    filtered_las_file_path = las_file_path.replace('.las', '_annotation.las')
    save_points_to_las(points_semantics, colors, filtered_las_file_path)

    print(f"Filtered LAS file saved to {filtered_las_file_path}")
    

if __name__ == "__main__":
    las_file_path = 'data/mission_a_0_density_sor_annotation.las'
    annotate_rock(las_file_path)
    #csv_file_path = 'data/pc_objects/mission_a_0_density_sor.csv'
    #filter_las_with_annotations(las_file_path, csv_file_path)
  