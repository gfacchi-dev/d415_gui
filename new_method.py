import open3d as o3d
import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import cv2 
from geometry import *
from camera import Camera
import torch
import math
from utils import *
from scipy.optimize import minimize
from qreader import QReader
import copy


def flip_180(pcd):
    """
    Flip the pointcloud of 180 degrees around x axis

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud

    Returns
    -------
    None.
    """
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
def rotate_90(pcd, clockwise=False) -> None:
    # Rotate of 90 degrees around z axis
    if clockwise:
        pcd.rotate([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    else:
        pcd.rotate([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

def rotation_matrix_from_axis_angle(axis, angle):
    # Crea la matrice di rotazione 3D da un asse e un angolo in radianti
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    rotation_matrix = np.array([
        [a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a**2 + c**2 - b**2 - d**2, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a**2 + d**2 - b**2 - c**2]
    ])

    return rotation_matrix

def angle_between_lines(m1, m2):
    # Calcola i vettori direzionali delle rette
    v1 = [1, m1]
    v2 = [1, m2]

    # Calcola il prodotto scalare e le norme dei vettori
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a**2 for a in v1))
    norm_v2 = math.sqrt(sum(b**2 for b in v2))

    # Calcola l'angolo tra i due vettori utilizzando l'arco coseno
    angle_rad = math.acos(dot_product / (norm_v1 * norm_v2))

    # Converti l'angolo da radianti a gradi, se necessario
    angle_deg = math.degrees(angle_rad)

    return angle_rad, angle_deg


def project_point_on_plane(x, y, z, A, B, C, D):
    denominatore = A**2 + B**2 + C**2
    xp = x - (A*x + B*y + C*z + D) * A / denominatore
    yp = y - (A*x + B*y + C*z + D) * B / denominatore
    zp = z - (A*x + B*y + C*z + D) * C / denominatore
    
    return xp, yp, zp

def filter_color(rgb, depth, color):
    # Definisci i range di colore rosso
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    if color=="red":
        lower_lower_red = np.array([170, 100, 100])
        upper_lower_red = np.array([180, 255, 255])
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_lower_red, upper_lower_red)
        mask += cv2.inRange(hsv, lower_red, upper_red)
    if color=="blue":
        lower_blue = np.array([90, 80, 80])
        upper_blue = np.array([150, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    rgb_filt = cv2.bitwise_and(rgb, rgb, mask=mask)
    depth_filt = cv2.bitwise_and(depth, depth, mask=mask)
    return rgb_filt, depth_filt







SCALA = 3
colorizer = rs.colorizer()
center = Camera("Center", "210622061176", SCALA, 100, 800)
left = Camera("Left", "211222063114", SCALA, 100, 800)
right = Camera("Right", "211122060792", SCALA, 100, 800)
cnt = 0
while True:
    if cnt<30:
        cnt += 1
        continue
    
    for i in range(0,90):
        center.get_frame_and_add_to_queue()
        left.get_frame_and_add_to_queue()
        right.get_frame_and_add_to_queue()
    
    (
        mean_left,
        variance_left,
        inliers_left,
        median_left
    ) = mean_variance_and_inliers_along_third_dimension_ignore_zeros(
        torch.from_numpy(left.depth_queue)
    )
    (
        mean_center,
        variance_center,
        inliers_center,
        median_center
    ) = mean_variance_and_inliers_along_third_dimension_ignore_zeros(
        torch.from_numpy(center.depth_queue)
    )
    (
        mean_right,
        variance_right,
        inliers_right,
        median_right
    ) = mean_variance_and_inliers_along_third_dimension_ignore_zeros(
        torch.from_numpy(right.depth_queue)
    )

    pcd_center = center.get_pcd_from_rgb_depth(center.last_rgb, mean_center)
    pcd_left = left.get_pcd_from_rgb_depth(left.last_rgb, mean_left)
    pcd_right = right.get_pcd_from_rgb_depth(right.last_rgb, mean_right)

    original_center = copy.deepcopy(pcd_center)
    original_left = copy.deepcopy(pcd_left)
    original_right = copy.deepcopy(pcd_right)

    # Flip it, otherwise the pointcloud will be upside down
    flip_180(pcd_center)
    flip_180(pcd_left)
    flip_180(pcd_right)

    # Rotate of 90 degrees around z axis
    rotate_90(pcd_center, clockwise=False)
    rotate_90(pcd_left, clockwise=True)
    rotate_90(pcd_right, clockwise=True)

    # Show pcd
    #o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd_center.points)
    a_c, b_c, c_c, d_c = fit_plane_to_points(points)
    plane_c = (a_c, b_c, c_c, d_c)
    n_c = (a_c, b_c, c_c)
    points = np.asarray(pcd_left.points)
    a_l, b_l, c_l, d_l = fit_plane_to_points(points)
    plane_l = (a_l, b_l, c_l, d_l)
    n_l = (a_l, b_l, c_l)
    points = np.asarray(pcd_right.points)
    a_r, b_r, c_r, d_r = fit_plane_to_points(points)
    plane_r = (a_r, b_r, c_r, d_r)
    n_r = (a_r, b_r, c_r)
    # REFERENCE PLANE
    p_ref = [0., 0., 1., 0.]


    # Draw plane 
    center_plane = get_plane_pcd(plane_c, color=[1, 0, 0])
    left_plane = get_plane_pcd(plane_l, color=[0, 1, 0])
    right_plane = get_plane_pcd(plane_r, color=[0, 0, 1])
    ref_plane = get_plane_pcd(p_ref, color=[0, 0, 0])

    
    o3d.visualization.draw_geometries([pcd_left, left_plane, pcd_right, right_plane, pcd_center, center_plane])

    # Calcola la matrice di rotazione tra i piani
    R_left = compute_rotation(plane_l, p_ref)
    R_right = compute_rotation(plane_r, p_ref)
    R_center = compute_rotation(plane_c, p_ref)

    pcd_left.rotate(R_left, center=(0,0,0))
    left_plane.rotate(R_left, center=(0,0,0))
    o3d.visualization.draw_geometries([pcd_left, left_plane, ref_plane])
    
    #t_a, t_b, t_c, d = fit_plane_to_points(np.asarray(pcd_left.points))
    _, _, _, d = fit_plane_to_points(np.asarray(left_plane.points))
    _, _, _, d2 = fit_plane_to_points(np.asarray(pcd_left.points))
    # print("d_pcd: ", d, " d_plane: ", d2)
    pcd_left.translate([0,0,-d2], relative=True)
    left_plane.translate([0,0,-d2], relative=True)
    o3d.visualization.draw_geometries([pcd_left, left_plane, ref_plane])

    pcd_right.rotate(R_right, center=(0,0,0))
    right_plane.rotate(R_right, center=(0,0,0))
    _, _, _, d = fit_plane_to_points(np.asarray(right_plane.points))
    _, _, _, d2 = fit_plane_to_points(np.asarray(pcd_right.points))
    print("d_pcd: ", d2, " d_plane: ", d)
    pcd_right.translate([0,0,-d2], relative=True)
    right_plane.translate([0,0,-d2], relative=True)
    o3d.visualization.draw_geometries([pcd_right, right_plane, ref_plane])

    pcd_center.rotate(R_center)
    center_plane.rotate(R_center)
    _, _, _, d = fit_plane_to_points(np.asarray(center_plane.points))
    _, _, _, d2 = fit_plane_to_points(np.asarray(pcd_center.points))
    print("d_pcd: ", d2, " d_plane: ", d)
    pcd_center.translate([0,0,-d2], relative=True)
    center_plane.translate([0,0,-d2], relative=True)
    o3d.visualization.draw_geometries([pcd_center, center_plane, ref_plane])

    o3d.visualization.draw_geometries([left_plane, center_plane, right_plane])
    o3d.visualization.draw_geometries([pcd_left, pcd_center, pcd_right])

    image_c = center.last_rgb
    image_l = left.last_rgb
    image_r = right.last_rgb

    # Decodifica i codici QR nell'immagine
    qreader = QReader()
    print(image_c.shape)
    center_qr = qreader.detect(image_c)[0]['quad_xy']
    print("center_qr: ", center_qr)
    corner_center_1 = np.array([center_qr[2][0], center_qr[2][1]], dtype=np.int32)
    corner_center_2 = np.array([center_qr[3][0], center_qr[3][1]], dtype=np.int32)

    left_qr = qreader.detect(image_l)[0]['quad_xy']
    print("left_qr: ", left_qr)
    corner_left_1 = np.array([left_qr[0][0], left_qr[0][1]], dtype=np.int32)
    corner_left_2 = np.array([left_qr[1][0], left_qr[1][1]], dtype=np.int32)

    right_qr = qreader.detect(image_r)[0]['quad_xy']
    print("right_qr: ", right_qr)
    corner_right_1 = np.array([right_qr[0][0], right_qr[0][1]], dtype=np.int32)
    corner_right_2 = np.array([right_qr[1][0], right_qr[1][1]], dtype=np.int32)

    print("Pixel 1 center: ", corner_center_1)
    corner_c1_index = rgb_point_to_pcd_index(corner_center_1, image_c, np.asarray(mean_center), center, original_center)
    print("Pixel 2 center: ", corner_center_2)
    corner_c2_index = rgb_point_to_pcd_index(corner_center_2, image_c, np.asarray(mean_center), center, original_center)

    print("Pixel 1 left: ", corner_left_1)
    corner_l1_index = rgb_point_to_pcd_index(corner_left_1, image_l, np.asarray(mean_left), left, original_left)
    print("Pixel 2 left: ", corner_left_2)
    corner_l2_index = rgb_point_to_pcd_index(corner_left_2, image_l, np.asarray(mean_left), left, original_left)

    print("Pixel 1 right: ", corner_right_1)
    corner_r1_index = rgb_point_to_pcd_index(corner_right_1, image_r, np.asarray(mean_right), right, original_right)
    print("Pixel 2 right: ", corner_right_2)
    corner_r2_index = rgb_point_to_pcd_index(corner_right_2, image_r, np.asarray(mean_right), right, original_right)

    # Trovo il punto corrispondente all'indice nella pointcloud
    corner_c_1 = np.asarray(pcd_center.points)[corner_c1_index][0]
    corner_c_2 = np.asarray(pcd_center.points)[corner_c2_index][0]
    corner_l_1 = np.asarray(pcd_left.points)[corner_l1_index][0]
    corner_l_2 = np.asarray(pcd_left.points)[corner_l2_index][0]
    corner_r_1 = np.asarray(pcd_right.points)[corner_r1_index][0]
    corner_r_2 = np.asarray(pcd_right.points)[corner_r2_index][0]

    print("corner_c_1: ", corner_c_1)
    print("corner_l_1: ", corner_l_1)
    print("corner_r_1: ", corner_r_1)
    T_L2C = get_t_from_correspondence(corner_l_1, corner_c_1)
    T_R2C = get_t_from_correspondence(corner_r_1, corner_c_1)
    print("T_L2C: ", T_L2C)
    print("T_R2C: ", T_R2C)
    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right])

    pcd_left.translate(T_L2C, relative=True)
    pcd_right.translate(T_R2C, relative=True)

    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right])


    v1 = corner_l_2 - corner_l_1
    v2 = corner_c_2 - corner_c_1
    R_L2C = matrix_between_vectors(v2, v1, clockwise=True)
    rot_center = np.asarray(pcd_left.points)[corner_l1_index][0]
    pcd_left.rotate(R_L2C, center=rot_center)
    
    o3d.visualization.draw_geometries([pcd_center, pcd_left])
    

    v1 = corner_r_1 - corner_r_2
    v2 = corner_c_1 - corner_c_2
    R_R2C = matrix_between_vectors(v2, v1, clockwise=False)
    rot_center = np.asarray(pcd_right.points)[corner_r1_index][0]
    pcd_right.rotate(R_R2C, center=rot_center)

    o3d.visualization.draw_geometries([pcd_center, pcd_right])

    print("R_L2C: ", R_L2C)
    print("T_L2C: ", T_L2C)
    print("R_R2C: ", R_R2C)
    print("T_R2C: ", T_R2C)
    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right])

    

