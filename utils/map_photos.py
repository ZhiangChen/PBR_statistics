import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import folium
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')


def extract_gps_data(image_path):
    """
    Extract GPS coordinates from image EXIF data.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if no GPS data found
    """
    try:
        with Image.open(image_path) as image:
            exif_data = image._getexif()
            
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "GPSInfo":
                    gps_data = {}
                    for gps_tag in value:
                        gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                        gps_data[gps_tag_name] = value[gps_tag]
                    
                    # Extract latitude and longitude
                    lat = gps_data.get("GPSLatitude")
                    lat_ref = gps_data.get("GPSLatitudeRef")
                    lon = gps_data.get("GPSLongitude")
                    lon_ref = gps_data.get("GPSLongitudeRef")
                    
                    if lat and lon and lat_ref and lon_ref:
                        # Convert DMS to decimal degrees - handle both tuple and fraction formats
                        def convert_to_decimal(dms_tuple):
                            degrees = float(dms_tuple[0])
                            minutes = float(dms_tuple[1])
                            seconds = float(dms_tuple[2])
                            return degrees + minutes/60 + seconds/3600
                        
                        lat_decimal = convert_to_decimal(lat)
                        lon_decimal = convert_to_decimal(lon)
                        
                        # Apply direction
                        if lat_ref == "S":
                            lat_decimal = -lat_decimal
                        if lon_ref == "W":
                            lon_decimal = -lon_decimal
                            
                        return lat_decimal, lon_decimal
    except Exception as e:
        print(f"Error reading GPS data from {image_path}: {e}")
    
    return None, None


def get_photo_locations(photo_folder):
    """
    Extract GPS coordinates from all photos in a folder and its subfolders.
    
    Args:
        photo_folder (str): Path to folder containing photos
        
    Returns:
        list: List of dictionaries containing photo info and coordinates
    """
    photo_data = []
    
    # Common image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp')
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(photo_folder):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(root, filename)
                lat, lon = extract_gps_data(image_path)
                
                if lat is not None and lon is not None:
                    # Get relative path from the main folder for better identification
                    relative_path = os.path.relpath(image_path, photo_folder)
                    photo_data.append({
                        'filename': filename,
                        'relative_path': relative_path,
                        'path': image_path,
                        'latitude': lat,
                        'longitude': lon
                    })
                else:
                    print(f"No GPS data found for {os.path.relpath(image_path, photo_folder)}")
    
    return photo_data


def create_photo_map_matplotlib(photo_data, output_path='photo_map.png', 
                               thumbnail_size=50, basemap_source='satellite'):
    """
    Create a map with photos plotted on satellite imagery using matplotlib and contextily.
    
    Args:
        photo_data (list): List of photo dictionaries with coordinates
        output_path (str): Path to save the output map
        thumbnail_size (int): Size of photo thumbnails in pixels
        basemap_source (str): Type of basemap ('satellite', 'osm', 'stamen_terrain')
    """
    if not photo_data:
        print("No photo data provided!")
        return
    
    # Create a DataFrame with photo locations
    df = pd.DataFrame(photo_data)
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Reproject to Web Mercator for use with contextily
    gdf = gdf.to_crs('EPSG:3857')
    
    # Calculate bounds with some padding
    bounds = gdf.total_bounds
    padding = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
    west, south, east, north = bounds[0] - padding, bounds[1] - padding, bounds[2] + padding, bounds[3] + padding
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Add basemap
    basemap_sources = {
        'satellite': ctx.providers.Esri.WorldImagery,
        'osm': ctx.providers.OpenStreetMap.Mapnik,
        'terrain': ctx.providers.Esri.WorldTopoMap,
        'cartodb': ctx.providers.CartoDB.Positron
    }
    
    try:
        ctx.add_basemap(ax, 
                       crs=gdf.crs.to_string(),
                       source=basemap_sources.get(basemap_source, ctx.providers.Esri.WorldImagery),
                       zoom='auto')
    except Exception as e:
        print(f"Warning: Could not load basemap. Error: {e}")
        print("Proceeding without basemap...")
    
    # Plot photo locations as dots
    gdf.plot(ax=ax, color='red', markersize=100, alpha=0.7, edgecolor='white', linewidth=2)
    
    # Add photo labels near the dots
    for idx, row in gdf.iterrows():
        try:
            x, y = row.geometry.x, row.geometry.y
            
            # Add filename as label near the dot (use relative path if different from filename)
            display_name = row.get('relative_path', row['filename'])
            # If relative path is just the filename, use filename only
            if display_name == row['filename']:
                display_name = row['filename']
            
            ax.annotate(display_name, 
                       xy=(x, y), 
                       xytext=(10, 10),
                       xycoords='data',
                       textcoords='offset points',
                       fontsize=4,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'),
                       ha='left',
                       va='bottom')
            
        except Exception as e:
            print(f"Error adding label for {row['filename']}: {e}")
    
    # Set bounds
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    plt.title(f'Photo Locations Map ({len(photo_data)} photos)', fontsize=16, fontweight='bold')
    
    # Add scale and north arrow (optional)
    ax.text(0.02, 0.98, 'N ↑', transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Map saved to {output_path}")
    
    plt.show()


def create_interactive_folium_map(photo_data, output_path='interactive_photo_map.html'):
    """
    Create an interactive map with photos using Folium.
    
    Args:
        photo_data (list): List of photo dictionaries with coordinates
        output_path (str): Path to save the HTML map
    """
    if not photo_data:
        print("No photo data provided!")
        return
    
    # Calculate center point
    center_lat = np.mean([photo['latitude'] for photo in photo_data])
    center_lon = np.mean([photo['longitude'] for photo in photo_data])
    
    # Create base map with no default tiles
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=18,
        max_zoom=25,
        tiles=None
    )
    
    # Add Google Hybrid as the first layer (this becomes the default)
    google_hybrid = folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google Hybrid',
        name='Google Hybrid',
        overlay=False,
        control=True,
        max_zoom=25
    )
    google_hybrid.add_to(m)
    
    # Add alternative tile layers
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        max_zoom=25
    ).add_to(m)
    
    # Add Esri Satellite as alternative
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
        name='Esri Satellite',
        max_zoom=25
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community',
        name='Esri World Topo',
        max_zoom=25
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB Positron',
        name='CartoDB Positron',
        max_zoom=25
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Set the satellite layer as the default by making it the first and active layer
    # This ensures it's selected when the map loads
    
    # Add photo markers
    for photo in photo_data:
        # Use relative path for display if available, otherwise use filename
        display_name = photo.get('relative_path', photo['filename'])
        
        # Create popup with image
        popup_html = f"""
        <div style="width: 200px;">
            <img src="file://{photo['path']}" width="180" height="120" style="object-fit: cover;">
            <br><b>{display_name}</b>
            <br>Lat: {photo['latitude']:.6f}
            <br>Lon: {photo['longitude']:.6f}
        </div>
        """
        
        folium.Marker(
            location=[photo['latitude'], photo['longitude']],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=display_name,
            icon=folium.Icon(color='red', icon='camera')
        ).add_to(m)
    
    # Save map
    m.save(output_path)
    print(f"Interactive map saved to {output_path}")
    
    return m


def plot_photos_on_map(photo_folder, map_type='both', output_dir='utils', 
                      thumbnail_size=50, basemap_source='satellite'):
    """
    Main function to plot photos with GPS coordinates on a map.
    
    Args:
        photo_folder (str or list): Path to folder containing photos, or list of photo file paths
        map_type (str): Type of map to create ('matplotlib', 'folium', or 'both')
        output_dir (str): Directory to save output files
        thumbnail_size (int): Size of thumbnails for matplotlib map
        basemap_source (str): Basemap source for matplotlib map
    """
    # Handle input - folder path or list of files
    if isinstance(photo_folder, str):
        if os.path.isdir(photo_folder):
            photo_data = get_photo_locations(photo_folder)
        else:
            print(f"Error: {photo_folder} is not a valid directory")
            return
    elif isinstance(photo_folder, list):
        # Assume it's a list of file paths
        photo_data = []
        for photo_path in photo_folder:
            if os.path.isfile(photo_path):
                lat, lon = extract_gps_data(photo_path)
                if lat is not None and lon is not None:
                    photo_data.append({
                        'filename': os.path.basename(photo_path),
                        'path': photo_path,
                        'latitude': lat,
                        'longitude': lon
                    })
            else:
                print(f"Warning: {photo_path} is not a valid file")
    else:
        print("Error: photo_folder must be a directory path or list of file paths")
        return
    
    if not photo_data:
        print("No photos with GPS data found!")
        return
    
    print(f"Found {len(photo_data)} photos with GPS coordinates")
    
    # Print coordinate bounds
    lats = [photo['latitude'] for photo in photo_data]
    lons = [photo['longitude'] for photo in photo_data]
    print(f"Latitude range: {min(lats):.6f} to {max(lats):.6f}")
    print(f"Longitude range: {min(lons):.6f} to {max(lons):.6f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create maps based on type requested
    if map_type in ['matplotlib', 'both']:
        matplotlib_output = os.path.join(output_dir, 'photo_map_matplotlib.png')
        create_photo_map_matplotlib(photo_data, matplotlib_output, 
                                   thumbnail_size, basemap_source)
    
    if map_type in ['folium', 'both']:
        folium_output = os.path.join(output_dir, 'photo_map_interactive.html')
        create_interactive_folium_map(photo_data, folium_output)
    
    return photo_data


# Example usage:
if __name__ == "__main__":
    # Example 1: Using a folder path
    photo_folder = "/Volumes/Extreme SSD/zhiang/manual"
    plot_photos_on_map(photo_folder, map_type='both', output_dir='utils')
    
    # Example 2: Using a list of photo paths
    # photo_paths = [
    #     "/path/to/photo1.jpg",
    #     "/path/to/photo2.jpg",
    #     "/path/to/photo3.jpg"
    # ]
    # plot_photos_on_map(photo_paths, map_type='both')
    
    # Example 3: Customized settings
    # plot_photos_on_map(
    #     photo_folder="/path/to/photos",
    #     map_type='matplotlib',
    #     output_dir='./maps',
    #     thumbnail_size=80,
    #     basemap_source='satellite'
    # )
    
    print("Photo mapping script ready!")
    print("\nTo use this script:")
    print("1. Call plot_photos_on_map() with your photo folder or list of photo paths")
    print("2. The script will extract GPS coordinates from EXIF data")
    print("3. Generate maps with photo locations and thumbnails")
    print("\nExample:")
    print("photo_data = plot_photos_on_map('/path/to/your/photos')")