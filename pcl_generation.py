import numpy as np 
import open3d as o3d

# CAD_path = "data/curve_without_map_latest_gt.ply"
CAD_path = 'ImageToStl.com_target_5_3.ply'
save_path = "target_5_3_ray.ply"

# cam_pos = [-300, 100, 500]
# cam_pos = [200,500,200]
cam_pos = [1,1500,1]
ray_ord = [0, 1, 0]

mesh = o3d.io.read_triangle_mesh(CAD_path)
mesh.compute_vertex_normals()

mesh_2 = o3d.geometry.TriangleMesh.create_sphere(radius=10).translate(cam_pos)
mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
mesh_t.compute_vertex_normals()

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
camera_view_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50).translate(cam_pos)

scene = o3d.t.geometry.RaycastingScene()
mesh_id = scene.add_triangles(mesh_t)

camera_look_at = mesh.get_axis_aligned_bounding_box().get_center()


'''
rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    fov_deg=90,
    center=camera_look_at,
    eye=cam_pos,
    up=ray_ord,
    width_px=1280*4,
    height_px=960*4,
)
'''
z= cam_pos[1]
rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    fov_deg=90,
    # center=camera_look_at,
    center=cam_pos - np.array([0, z, 1]),
    eye=cam_pos,
    up=ray_ord,
    width_px=1280,
    height_px=960,
    # width_px=1000,
    # height_px=900,

)




# Original
ans = scene.cast_rays(rays)
hit = ans['t_hit'].isfinite()
points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
pcd = o3d.t.geometry.PointCloud(points)
o3d.visualization.draw_geometries([pcd.to_legacy(), mesh_2, axis, camera_view_mesh])
o3d.io.write_point_cloud(save_path, pcd.to_legacy())
