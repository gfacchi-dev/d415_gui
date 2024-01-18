import numpy as np
from geometry import *
from utils import *
import open3d as o3d

# Esempio di utilizzo
punto = np.array([0.2, 0.2, 0.2])
piano = np.array([-0.186, 0.045, 1, 2.0884])  # Piano z=0


proiezione = project_point_on_plane(punto, piano)
print("Punto originale:", punto)
print("Proiezione sul piano:", proiezione)

plane_pcd = get_plane_pcd(piano)
# COmpute vertex normals
plane_pcd.estimate_normals()
# point_pcd = o3d.geometry.PointCloud()
# point_pcd.points = o3d.utility.Vector3dVector([punto, proiezione])
# point_pcd.paint_uniform_color([1, 0, 0])
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=200)
sphere.paint_uniform_color([0, 1, 0])
sphere.translate(punto, relative=False)

proj_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=200)
proj_sphere.paint_uniform_color([1, 0, 0])
proj_sphere.translate(proiezione, relative=False)

reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
o3d.visualization.draw_geometries([plane_pcd, reference_frame, sphere, proj_sphere])
print(punto-proiezione)