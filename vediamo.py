import cv2
import numpy as np
from camera import Camera
from geometry import *

p1 = [1.2, 3.2, 2.2, 3.]
p2 = [1.2, 3.2, 2.2, 4.]
pref = [0., 0., 1., 0.]
pref2 = [0., 0., 1., 1.]

pcd1 = get_plane_pcd(p1, color=[0, 0, 1])
pcd2 = get_plane_pcd(p2, color=[1, 0, 0])
pcd_ref = get_plane_pcd(pref, color=[0, 1, 0])
pcd_ref2 = get_plane_pcd(pref2, color=[0, 1, 0])

o3d.visualization.draw_geometries([pcd1, pcd_ref])
R = compute_rotation(p1, pref)
_, _, _, D1 = fit_plane_to_points(np.asarray(pcd1.points))
t = np.array([0, 0, -D1])
t_rot = np.dot(R, t)

# Matrice di trasformazione 4x4 da applicare a pcd1
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t_rot

pcd1.transform(T)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

o3d.visualization.draw_geometries([pcd1, pcd_ref, mesh_frame])

pcd2.transform(T)
o3d.visualization.draw_geometries([pcd2, pcd_ref2, mesh_frame])