import numpy as np
from sklearn.linear_model import LinearRegression
import scipy
import open3d as o3d
from scipy.optimize import minimize


def remove_outliers(point_cloud, threshold=1.0):
    # Calcola la media e la deviazione standard per ciascuna dimensione
    mean_values = np.mean(point_cloud, axis=0)
    std_deviation = np.std(point_cloud, axis=0)

    # Calcola la distanza euclidea di ciascun punto dalla media
    distances = np.linalg.norm(point_cloud - mean_values, axis=1)

    # Identifica gli indici degli outliers in base alla soglia specificata
    outlier_indices = np.where(distances > threshold * np.max(std_deviation))

    # Rimuovi gli outliers dalla point cloud
    filtered_point_cloud = np.delete(point_cloud, outlier_indices, axis=0)
    
    return filtered_point_cloud

def fit_plane_to_points(points):
    points = np.array(points)
    points = remove_outliers(points)

    # Salva i points filtrati in un np array e ritornalo in formato open3d pointcloud
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(points)

    ones_column = np.ones((points.shape[0], 1))
    # Keep only first 2 dimensions of points
    points_extended = np.hstack([points[:, :2], ones_column])
    coeff, _, _, _ = np.linalg.lstsq(points_extended, points[:, -1], rcond=None)
    A, B, D = coeff
    C = -1
    #est_d = (- A * points[:, 0] - B * points[:, 1] - C * points[:, 2])
    #D = np.mean(est_d)

    return -A, -B, -C, -D#, o3d_points


def get_rotation_between_lines(v1, v2) -> np.ndarray((3,3)):
    # Passo 1
    k = np.array([0, 0, 1])  # Asse z
    n1 = np.cross(v1, k)
    n2 = np.cross(v2, k)

    # Passo 2
    n1_norm = n1 / np.linalg.norm(n1)
    n2_norm = n2 / np.linalg.norm(n2)

    # Passo 3
    theta = np.arccos(np.dot(n1_norm, n2_norm))

    # Passo 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])

    return R


def compute_rotation(p1, p2):
    A1, B1, C1, D1 = p1
    A2, B2, C2, D2 = p2
    Z =  (-A1 * 1 - B1 * 1 - D1) / C1
    # Normalizza i vettori normali
    n1 = np.array([A1, B1, C1])
    n2 = np.array([A2, B2, C2])
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    # Trova l'asse di rotazione
    asse_rotazione = np.cross(n1, n2)
    asse_rotazione /= np.linalg.norm(asse_rotazione)
    # Trova l'angolo di rotazione
    angolo_rotazione = np.arccos(np.dot(n1, n2))
    # Matrice di rotazione
    R = np.array([
        [np.cos(angolo_rotazione) + asse_rotazione[0]**2 * (1 - np.cos(angolo_rotazione)),
         asse_rotazione[0] * asse_rotazione[1] * (1 - np.cos(angolo_rotazione)) - asse_rotazione[2] * np.sin(angolo_rotazione),
         asse_rotazione[0] * asse_rotazione[2] * (1 - np.cos(angolo_rotazione)) + asse_rotazione[1] * np.sin(angolo_rotazione)],
        [asse_rotazione[1] * asse_rotazione[0] * (1 - np.cos(angolo_rotazione)) + asse_rotazione[2] * np.sin(angolo_rotazione),
         np.cos(angolo_rotazione) + asse_rotazione[1]**2 * (1 - np.cos(angolo_rotazione)),
         asse_rotazione[1] * asse_rotazione[2] * (1 - np.cos(angolo_rotazione)) - asse_rotazione[0] * np.sin(angolo_rotazione)],
        [asse_rotazione[2] * asse_rotazione[0] * (1 - np.cos(angolo_rotazione)) - asse_rotazione[1] * np.sin(angolo_rotazione),
         asse_rotazione[2] * asse_rotazione[1] * (1 - np.cos(angolo_rotazione)) + asse_rotazione[0] * np.sin(angolo_rotazione),
         np.cos(angolo_rotazione) + asse_rotazione[2]**2 * (1 - np.cos(angolo_rotazione))]
    ])

    return R



def get_plane_pcd(coeffs, color: list = [0, 0, 1]):
    # Estrai i coefficienti
    A, B, C, D = coeffs

    # Calcola i punti del piano
    x = np.linspace(-1, 1, 300)
    y = np.linspace(-1, 1, 300)
    X, Y = np.meshgrid(x, y)
    Z = (- A * X - B * Y - D) / C
    # Converti in pointcloud
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    # Create Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color(color)
    return point_cloud


def error_function(params, data_points):
    P0 = params[:3]  # i primi tre valori sono le coordinate di P0
    v = params[3:]   # i successivi tre valori sono le componenti del vettore direzionale
    distances = np.linalg.norm(np.cross(data_points - P0, v), axis=1) / np.linalg.norm(v)
    return np.sum(distances**2)

def fit_line(data_points):
    # Inizializzazione dei parametri (sostituisci con valori ragionevoli)
    initial_params = np.array([0,1,0, 0,0.5,0])

    # Ottimizzazione dei parametri utilizzando la libreria scipy
    result = minimize(error_function, initial_params, args=(data_points,), method='L-BFGS-B')

    # Estrazione dei parametri ottimali
    optimal_params = result.x
    P0_optimal = optimal_params[:3]
    v_optimal = optimal_params[3:]

    return P0_optimal, v_optimal

def get_t_from_correspondence(p1, p1_c):
    x1, y1 = p1[:2]
    x1_c, y1_c = p1_c[:2]

    Dx = - (x1 - x1_c)
    Dy = - (y1 - y1_c)
    return np.array([Dx, Dy, 0])

def get_R_from_correspondence(p1_l, p1_r, p2_l, p2_r):
    x1_l, y1_l = p1_l[:2]
    x2_l, y2_l = p2_l[:2]
    x1_r, y1_r = p1_r[:2]
    x2_r, y2_r = p2_r[:2]
    # Trova l'angolo di rotazione
    theta_l = np.arctan2(y2_l - y1_l, x2_l - x1_l)
    theta_r = np.arctan2(y2_r - y1_r, x2_r - x1_r)
    theta = theta_l - theta_r
    print("Angolo:" ,theta)

    # Costruisci la matrice di rotazione
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return rotation_matrix

import cv2
def rgb_point_to_pcd_index(point, rgb_frame, depth_frame, camera, original_point_cloud):
    print(rgb_frame.shape)
    print(depth_frame.shape)
    depth_masked_point = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=np.uint16)
    depth_masked_point[point[1], point[0]] = depth_frame[point[1], point[0]]
    print(depth_masked_point.shape)
    print(type(depth_masked_point))
    tmp = rgb_frame.copy()
    cv2.circle(tmp, (point[0], point[1]), 5, (255,0,0), -1)
    cv2.imshow("tmp", tmp)
    cv2.waitKey(0)
    pcd_corner_c_1 = camera.get_pcd_from_rgb_depth(rgb_frame, depth_masked_point)
    point_coord = np.asarray(pcd_corner_c_1.points)[0]
    points = np.asarray(original_point_cloud.points)
    index = np.where(np.all(points == point_coord, axis=1))[0]
    return index

def matrix_between_vectors(v1, v2, clockwise: bool = True):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Calcola l'angolo in radianti
    if clockwise:
        angle_rad = - (np.arccos(dot_product / (norm_v1 * norm_v2)))
    else:
        angle_rad = (np.arccos(dot_product / (norm_v1 * norm_v2)))
    # Matrice di rotazione attorno all'asse z
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    return R

def project_point_on_plane(point, plane):
    a, b, c = plane[:3]
    d = plane[3]
    x, y, z = point

    den = a**2 + b**2 + c**2
    proj_x = x - ((a*x + b*y + c*z + d*c) / den) * a
    proj_y = y - ((a*x + b*y + c*z + d*c) / den) * b
    proj_z = z - ((a*x + b*y + c*z + d*c) / den) * c

    return np.array([proj_x, proj_y, proj_z])