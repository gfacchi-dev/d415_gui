import numpy as np
from geometry import *
import open3d as o3d
import matplotlib.pyplot as plt


def points_from_plane(plane, std=0.1, begin=10, stop=10, n_points=300, noise=True):
    A, B, C, D = plane
    x = np.linspace(-begin, stop, n_points)
    y = np.linspace(-begin, stop, n_points)
    x, y = np.meshgrid(x, y)
    Z = (- A * x - B * y - D) / C
    if noise:
        Z += np.random.normal(0, std, Z.shape)
        x += np.random.normal(0, std, x.shape)
        y += np.random.normal(0, std, y.shape)
    points = np.column_stack((x.flatten(), y.flatten(), Z.flatten()))
    return points

def points_from_line(line, std=0.1, begin=10, stop=10, n_points=300, noise=True):
    A, B, C = line
    x = np.linspace(-begin, stop, n_points)
    y = (- A * x - C ) / B
    if noise:
        x += np.random.normal(0, std, x.shape)
        y += np.random.normal(0, std, y.shape)
    points = np.column_stack((x.flatten(), y.flatten()))
    return points


# p_c = [0., 0.2, 1., 1.]
# p_l = [6., 2., 5., 4.]
p_c = [1., 2., 8., 1.]
p_l = [-1., 4., 5., 2.]

real_p_c = points_from_plane(p_c, std=0.01, begin=1, stop=1, n_points=300, noise=True)
real_p_l = points_from_plane(p_l, std=0.01, begin=1, stop=1, n_points=300, noise=True)

a,b,c,d = fit_plane_to_points(real_p_c)
est_p_c = [a,b,c,d]
est_pcd_c = o3d.geometry.PointCloud()
est_pcd_c.points = o3d.utility.Vector3dVector(points_from_plane(est_p_c, std=0.01, begin=1, stop=1, n_points=300, noise=False))
est_pcd_c.paint_uniform_color([0, 0, 1])
a,b,c,d = fit_plane_to_points(real_p_l)
est_p_l = [a,b,c,d]
est_pcd_l = o3d.geometry.PointCloud()
# est_pcd_l.normals = o3d.utility.Vector3dVector([est_p_l[:3]])
est_pcd_l.points = o3d.utility.Vector3dVector(points_from_plane(est_p_l, std=0.01, begin=1, stop=1, n_points=300, noise=False))

print("R plane_c: ", p_c)
print("E plane_c: ", est_p_c)
print("R plane_l: ", p_l)
print("E plane_l: ", est_p_l)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
# Create Open3D PointCloud
real_pcd_c = o3d.geometry.PointCloud()
real_pcd_c.points = o3d.utility.Vector3dVector(real_p_c)
real_pcd_c.paint_uniform_color([0, 0, 1])
real_pcd_l = o3d.geometry.PointCloud()
real_pcd_l.points = o3d.utility.Vector3dVector(real_p_l)
real_pcd_l.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([real_pcd_c, real_pcd_l, mesh_frame])

R_l, p_l_rot = compute_rotation(est_p_l, est_p_c)
real_pcd_l.rotate(R_l)
est_pcd_l.rotate(R_l)

o3d.visualization.draw_geometries([real_pcd_c, real_pcd_l, est_pcd_l, mesh_frame])

a,b,c,d = fit_plane_to_points(np.asarray(est_pcd_l.points))

t_vector = translation_vector([a,b,c,d], est_p_c)
#t_vector = translation_vector(est_p_c, [a,b,c,d])

#distance = np.linalg.norm(t_vector)
#t_vector = distance * np.array([a,b,c])

est_pcd_l.translate(t_vector, relative=True)
real_pcd_l.translate(t_vector, relative=True)
print(f'translation vector: {t_vector}')
o3d.visualization.draw_geometries([real_pcd_c, est_pcd_l, real_pcd_l, mesh_frame])