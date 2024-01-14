import open3d as o3d
import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import cv2 
from geometry import *
from camera import Camera
import torch
import math

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
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
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
    if cnt<270:
        cnt += 1
        continue
    pcd_center, (rgb_c, depth_c) = center.get_pcd_and_frames(filter_pcd=True)
    pcd_left, (rgb_l, depth_l) = left.get_pcd_and_frames(filter_pcd=True)
    pcd_right, (rgb_r, depth_r) = right.get_pcd_and_frames(filter_pcd=True)

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
    a_c, b_c, c_c, d_c, filtered_points_c = fit_plane_to_points(points)
    plane_c = (a_c, b_c, c_c, d_c)
    n_c = (a_c, b_c, c_c)
    points = np.asarray(pcd_left.points)
    a_l, b_l, c_l, d_l, filtered_points_l = fit_plane_to_points(points)
    plane_l = (a_l, b_l, c_l, d_l)
    n_l = (a_l, b_l, c_l)
    points = np.asarray(pcd_right.points)
    a_r, b_r, c_r, d_r, filtered_points_r = fit_plane_to_points(points)
    plane_r = (a_r, b_r, c_r, d_r)
    n_r = (a_r, b_r, c_r)

    o3d.visualization.draw_geometries([filtered_points_c, pcd_center])
    o3d.visualization.draw_geometries([filtered_points_l, pcd_left])
    o3d.visualization.draw_geometries([filtered_points_r, pcd_right])

    # Draw plane 
    center_plane = get_plane_pcd(plane_c, color=[1, 0, 0])
    left_plane = get_plane_pcd(plane_l, color=[0, 1, 0])
    right_plane = get_plane_pcd(plane_r, color=[0, 0, 1])

    o3d.visualization.draw_geometries([filtered_points_c, center_plane])
    o3d.visualization.draw_geometries([filtered_points_l, left_plane])
    o3d.visualization.draw_geometries([filtered_points_r, right_plane])
    
    o3d.visualization.draw_geometries([pcd_left, left_plane])
    o3d.visualization.draw_geometries([pcd_right, right_plane])

    o3d.visualization.draw_geometries([pcd_left, left_plane, pcd_center, center_plane, pcd_right, right_plane])
    # Riduci le cifre decimali con il print formattato

    print(f"Equazione del piano centrale: {a_c}x + {b_c}y + {c_c}z + {d_c} = 0")
    print(f"Equazione del piano sinistro: {a_l}x + {b_l}y + {c_l}z + {d_l} = 0")
    print(f"Equazione del piano destro: {a_r}x + {b_r}y + {c_r}z + {d_r} = 0")

    print(f"n_c: {n_c}")
    print(f"n_l: {n_l}")
    print(f"n_r: {n_r}")

    # Calcola la matrice di rotazione tra i piani
    R_left, plane_l_rot = compute_rotation(plane_l, plane_c)
    R_right, plane_r_rot = compute_rotation(plane_r, plane_c)

    pcd_left.rotate(R_left)
    pcd_right.rotate(R_right)

    distance_l2c = distance_between_planes(plane_l_rot, plane_c)
    distance_r2c = distance_between_planes(plane_r_rot, plane_c)

    n_c /= np.linalg.norm(n_c)
    n_l /= np.linalg.norm(n_l)
    n_r /= np.linalg.norm(n_r)


    translation_left = distance_l2c * n_l
    translation_right = distance_r2c * n_r

    pcd_left.translate(translation_left, relative=True)
    pcd_right.translate(translation_right, relative=True)

    o3d.visualization.draw_geometries([left_plane, center_plane, right_plane])
    continue
    #o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right, point_cloud])
    
    # Soglia per colore le due linee che costituiscono la croce 
    red_c = filter_color(rgb_c, depth_c, "red")
    red_l = filter_color(rgb_l, depth_l, "red")
    red_r = filter_color(rgb_r, depth_r, "red")

    blue_c = filter_color(rgb_c, depth_c, "blue")
    blue_l = filter_color(rgb_l, depth_l, "blue")
    blue_r = filter_color(rgb_r, depth_r, "blue")


    # Costruisco point cloud con i soli punti rossi
    red_pcd_c = c_c.get_pcd_from_rgb_depth(red_c, depth_c)
    red_pcd_l = c_l.get_pcd_from_rgb_depth(red_l, depth_l)
    red_pcd_r = c_r.get_pcd_from_rgb_depth(red_r, depth_r)

    blue_pcd_c = c_c.get_pcd_from_rgb_depth(blue_c, depth_c)
    blue_pcd_l = c_l.get_pcd_from_rgb_depth(blue_l, depth_l)
    blue_pcd_r = c_r.get_pcd_from_rgb_depth(blue_r, depth_r)


    proj_points_c = np.zeros((len(red_pcd_c.points), 3))
    proj_points_l = np.zeros((len(red_pcd_l.points), 3))
    proj_points_r = np.zeros((len(red_pcd_r.points), 3))

    # Proietto le pointcloud rosse sul piano stimato 
    for i, point in enumerate(red_pcd_c.points):
        point[0], point[1], point[2] = project_point_on_plane(point[0], point[1], point[2], a_c, b_c, c_c, d_c)
        proj_points_c[i] = point
    for i, point in enumerate(red_pcd_l.points):
        point[0], point[1], point[2] = project_point_on_plane(point[0], point[1], point[2], a_c, b_c, c_c, d_c)
        proj_points_l[i] = point
    for i, point in enumerate(red_pcd_r.points):
        point[0], point[1], point[2] = project_point_on_plane(point[0], point[1], point[2], a_c, b_c, c_c, d_c)
        proj_points_r[i] = point

    a_line_c_red, b_line_c_red, c_line_c_red, d_line_c_red = fit_line(proj_points_c)
    a_line_l_red, b_line_l_red, c_line_l_red, d_line_l_red = fit_line(proj_points_l)
    a_line_r_red, b_line_r_red, c_line_r_red, d_line_r_red = fit_line(proj_points_r)

    proj_points_c = np.zeros((len(blue_pcd_c.points), 3))
    proj_points_l = np.zeros((len(blue_pcd_l.points), 3))
    proj_points_r = np.zeros((len(blue_pcd_r.points), 3))
    # Proietto le pointcloud blu sul piano stimato
    for i, point in enumerate(blue_pcd_c.points):
        point[0], point[1], point[2] = project_point_on_plane(point[0], point[1], point[2], a_c, b_c, c_c, d_c)
        proj_points_c[i] = point
    for i, point in enumerate(blue_pcd_l.points):
        point[0], point[1], point[2] = project_point_on_plane(point[0], point[1], point[2], a_c, b_c, c_c, d_c)
        proj_points_l[i] = point
    for i, point in enumerate(blue_pcd_r.points):
        point[0], point[1], point[2] = project_point_on_plane(point[0], point[1], point[2], a_c, b_c, c_c, d_c)
        proj_points_r[i] = point

    a_line_c_blue, b_line_c_blue, c_line_c_blue, d_line_c_blue = fit_line(proj_points_c)
    a_line_l_blue, b_line_l_blue, c_line_l_blue, d_line_l_blue = fit_line(proj_points_l)
    a_line_r_blue, b_line_r_blue, c_line_r_blue, d_line_r_blue = fit_line(proj_points_r)

    intersection_c = find_intersection(a_line_c_red, b_line_c_red, c_line_c_red, d_line_c_red, a_line_c_blue, b_line_c_blue, c_line_c_blue, d_line_c_blue)
    intersection_l = find_intersection(a_line_l_red, b_line_l_red, c_line_l_red, d_line_l_red, a_line_l_blue, b_line_l_blue, c_line_l_blue, d_line_l_blue)
    intersection_r = find_intersection(a_line_r_red, b_line_r_red, c_line_r_red, d_line_r_red, a_line_r_blue, b_line_r_blue, c_line_r_blue, d_line_r_blue)


    translation_l_2_c = intersection_c - intersection_l
    translation_r_2_c = intersection_c - intersection_r

    # Trasla la pointcloud di sinistra
    pcd_left.translate(translation_l_2_c, relative=True)
    # Trasla la pointcloud di destra
    pcd_right.translate(translation_r_2_c, relative=True)

    # Calcola la matrice di rotazione le rette blu 
    m_line_c_blue = -a_line_c_blue / b_line_c_blue
    m_line_l_blue = -a_line_l_blue / b_line_l_blue
    m_line_r_blue = -a_line_r_blue / b_line_r_blue

    angle_l_blue, angle_l_blue_deg = angle_between_lines(m_line_c_blue, m_line_l_blue)
    angle_r_blue, angle_r_blue_deg = angle_between_lines(m_line_c_blue, m_line_r_blue)


    # Calcola la matrice di rotazione tra le rette rosse
    m_line_c_red = -a_line_c_red / b_line_c_red
    m_line_l_red = -a_line_l_red / b_line_l_red
    m_line_r_red = -a_line_r_red / b_line_r_red

    angle_l_red, angle_l_red_deg = angle_between_lines(m_line_c_red, m_line_l_red)
    angle_r_red, angle_r_red_deg = angle_between_lines(m_line_c_red, m_line_r_red)

    # Calcola la matrice di rotazione media tra le rette blu e rosse
    angle_l = (angle_l_blue + angle_l_red) / 2
    angle_r = (angle_r_blue + angle_r_red) / 2

    rotation_matrix_l_2_c = rotation_matrix_from_axis_angle([0, 0, 1], angle_l)
    rotation_matrix_r_2_c = rotation_matrix_from_axis_angle([0, 0, 1], angle_r)

    # Ruota la pointcloud di sinistra
    pcd_left.rotate(rotation_matrix_l_2_c)
    # Ruota la pointcloud di destra
    pcd_right.rotate(rotation_matrix_r_2_c)
    

    # TODO: Trovare la rotazione che allinei al meglio le linee
    # Per ogni vista allineare la linea orizzontale con la linea orizzontale della vista centrale
    # Calcolare la matrice di rotazione tra le due linee
    # Ruotare la pointcloud di sinistra e di destra

    o3d.visualization.draw_geometries([pcd_center, pcd_right])
    

