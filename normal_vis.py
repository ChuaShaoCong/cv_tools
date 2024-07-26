'''Visualise the normals of a mesh with a heat map based on the angle with the top view.'''
import numpy as np 
import copy
import open3d as o3d

import time
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# PARAMETERS
# CAD_path = "angle_validation_data/curved_validation_target2.ply"
CAD_path = "angle_validation_data/scaled_5_5.ply"
top_view_axis = np.array([0, 1, 0])  # Assuming top view is along the Y-axis
filter_percentile = 98 
hardcoded_max_angle = 35

def normalize_vector(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def angle_with_top_view(normal,top_view_axis=[0,1,0]):
    """Calculate the angle between a normal and the top view, handling flipped normals."""
    
    # Normalize the normal vector
    normal = normalize_vector(np.array(normal))
    
    # Since some normals are inverted, we need to consider both cases (normal and -normal)
    dot_product_original = np.dot(normal, top_view_axis)
    dot_product_negated = np.dot(-normal, top_view_axis)
    
    # Normalize vectors for calculating norms
    norm_vec1 = np.linalg.norm(normal)
    norm_vec2 = np.linalg.norm(top_view_axis)

    # Calculate cosine of angles for both cases
    cos_theta_original = dot_product_original / (norm_vec1 * norm_vec2)
    cos_theta_negated = dot_product_negated / (norm_vec1 * norm_vec2)

    # Clip values to ensure they are within valid range [-1, 1]
    cos_theta_original = np.clip(cos_theta_original, -1.0, 1.0)
    cos_theta_negated = np.clip(cos_theta_negated, -1.0, 1.0)

    # Calculate angles in radians for both cases
    theta_radians_original = np.arccos(cos_theta_original)
    theta_radians_negated = np.arccos(cos_theta_negated)

    # Convert angles to degrees
    theta_degrees_original = np.degrees(theta_radians_original)
    theta_degrees_negated = np.degrees(theta_radians_negated)
    
    # Take the minimum of the two angles to account for flipped normals
    min_angle_degrees = min(theta_degrees_original, theta_degrees_negated)
    
    return min_angle_degrees

# Heat map visualisation functions
def interpolate_color(d, d_min, d_max, color_start, color_end):
    ratio = (d - d_min) / (d_max - d_min)
    return [int(color_start[i] + ratio * (color_end[i] - color_start[i])) for i in range(3)]

def get_color_gradient(distances, max_distance_eval=4.0):
    np_colors = []
    for dist in distances:
        if dist <= max_distance_eval / 2:
            color = interpolate_color(dist, 0.0, max_distance_eval / 2, [0, 255, 0], [255, 255, 0])
        elif dist > max_distance_eval:
            color = [128, 128, 128]
        else:
            color = interpolate_color(dist, max_distance_eval / 2, max_distance_eval, [255, 255, 0], [255, 0, 0])
        color = np.array(color) / 255.0
        np_colors.append(color)
    return np.array(np_colors).astype(np.float32)

# Statistics and plotting functions
def calculate_statistics(res):
    print(f'mean error = {np.mean(res)}, std = {np.std(res)}')

def filter_outlier(res, filter_percentile):
    res_filtered = res.copy()
    # remove the top 20% of the data
    res_filtered = np.array(res_filtered)
    res_filtered = res_filtered[res_filtered < np.percentile(res_filtered, filter_percentile)]
    return res_filtered

def plot_histogram(res):
    res_pl = res
    mu, std = norm.fit(res_pl)

    n, bins, patches = plt.hist(res_pl, bins=20, density=True, alpha=0.8, color='g')

    vmax = np.percentile(res_pl, 80)
    normalizer = mpl.colors.Normalize(vmin=-0.1, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')

    for c, p in zip(bins, patches):
        plt.setp(p, 'facecolor', mapper.to_rgba(c))
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = f"Analysis results: mean = {mu:.2f},  std = {std:.2f}"
    plt.title(title)

    # label axis
    plt.xlabel('Normal Angle (deg)')
    plt.ylabel('Density')

    plt.show()


if __name__ == "__main__":
    # Load the mesh and estimate normals
    mesh = o3d.io.read_point_cloud(CAD_path)
    mesh.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    o3d.visualization.draw_geometries([mesh]) # TODO press N to visualise normals
    normals = mesh.normals

    # # If the input file is CAD STEP file, not from the ray tracing mesh genererated from pcl_generation.py
    # ''' For CAD, the point cloud is generated from the vertices of the mesh, and the normals are calculated for the point cloud. This results in a sparser point cloud which is less accurate'''
    # mesh = o3d.io.read_triangle_mesh(CAD_path)
    # mesh.compute_vertex_normals()
    # mesh_point_cloud = o3d.geometry.PointCloud()
    # mesh_point_cloud.points = mesh.vertices
    # mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # mesh_t.compute_vertex_normals()
    # # TODO For CAD not ray trace
    # normals = mesh_t.vertex.normals

    angle_degrees = []
    for normal in normals:
        # TODO for CAD not ray trace
        # normal = normal.numpy()

        angle = angle_with_top_view(normal)
        print(f"Angle: {angle}")
        angle_degrees.append(angle)

    # Find the maximum angle 
    max_angle = max(angle_degrees)
    plot_histogram(angle_degrees)
    print(f"Max angle: {max_angle}")

    # Remove outliers by removing the top % of the data
    angle_degrees_filtered = filter_outlier(angle_degrees, filter_percentile)
    plot_histogram(angle_degrees_filtered)
    max_angle_filtered = max(angle_degrees_filtered)
    print(f"Max angle filtered: {max_angle_filtered}")

    # Visualise the normals with a heat map
    # colors_rgb = get_color_gradient(angle_degrees, max_angle_filtered)
    colors_rgb = get_color_gradient(angle_degrees, hardcoded_max_angle)
    colors = np.array(colors_rgb, dtype=np.float32)
    vis_pcl = copy.deepcopy(mesh)
    vis_pcl.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([vis_pcl])
