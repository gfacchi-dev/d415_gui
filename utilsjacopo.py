import os 
import pickle
import numpy as np
import torch
from geometry import * 
import open3d as o3d
import copy
from qreader import QReader
import datetime


def check_calibration()-> bool:
    """
    Checks if the calibration file is present in the calibration folder.
    :return: True if the file is present, False otherwise.
    """
    dirs = os.listdir("./calibrations")
    if len(dirs) == 0:
        return False
    else:
        return True

def get_calibration()-> tuple:
    """
    Returns the last calibration matrices.
    :return: (M_L2C, M_R2C)
    """
    dirs = os.listdir("./calibrations")
    dirs.sort(reverse=True)
    # TODO: check if the last calibration is enough recent
    return dirs[0]

def mean_variance_and_inliers_along_third_dimension_ignore_zeros(tensor: torch.Tensor):
    # Check if the input tensor has the expected shape
    if tensor.shape[-1] != 90:
        raise ValueError(
            "Input tensor should have shape (..., 90) for the third dimension."
        )
    
    non_zero_mask = tensor != 0
    num_non_zero_elements = non_zero_mask.sum(dim=2)
    nineties = 89 * torch.ones_like(num_non_zero_elements)
    print(nineties.shape)
    num_zero_values = nineties.sub(num_non_zero_elements)
    sorted_tensor = tensor.sort(dim=2)
    fortyfives = 45 * torch.ones_like(num_non_zero_elements)
    median_indexes = fortyfives.add(num_zero_values/2).type(torch.int64)
    medians = torch.gather(tensor, 2, median_indexes.unsqueeze(2).repeat(1,1,90))
    sum_values = tensor.sum(dim=2)
    mean_values = sum_values / num_non_zero_elements
    diff_squared = torch.where(
        non_zero_mask, 
        (tensor - mean_values.unsqueeze(2).repeat(1,1,90)) ** 2,
        torch.zeros_like(tensor)
    )
    variance_values = diff_squared.sum(dim=2)
    num_non_zero_elements_v = num_non_zero_elements
    variance_values = torch.div(variance_values, num_non_zero_elements_v)
    variance_values = torch.where(
        num_non_zero_elements != 0,
        variance_values,
        torch.sub(torch.zeros_like(num_non_zero_elements), 1)
    )

    return mean_values, variance_values, num_non_zero_elements, medians

def flip_180(pcd):
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

def rot_matrix_from_deg_angles(alpha, beta, gamma):
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def fit_func(params, A, B):
    alpha, beta, gamma = params[:3]
    t = params[3:]
    # Convert from degrees to radians
    R = rot_matrix_from_deg_angles(alpha, beta, gamma)
    transformed_A = np.dot(R, A) + t[:, np.newaxis]
    error = B - transformed_A
    return np.sum(error**2)

def find_rotation_translation_ransac(A, B, num_iterations=1000, tolerance=1e-6):
    best_params = None
    best_error = float('inf')

    for _ in range(num_iterations):
        # Seleziona un set casuale di punti
        indices = np.random.choice(A.shape[1], size=4, replace=False)
        sampled_A = A[:, indices]
        sampled_B = B[:, indices]

        # Utilizza i minimi quadrati per trovare la rototraslazione
        result = minimize(fit_func, x0=np.zeros(6), args=(sampled_A, sampled_B), method='Powell')

        # Calcola l'errore complessivo
        total_error = fit_func(result.x, A, B)

        # Aggiorna i migliori parametri se necessario
        if total_error < best_error:
            best_error = total_error
            best_params = result.x

        # Termina se l'errore è inferiore alla tolleranza
        if best_error < tolerance:
            print("Tolleranza raggiunta")
            break

    # Estrai la matrice di trasformazione dai migliori parametri
    R = rot_matrix_from_deg_angles(*best_params[:3])
    t = best_params[3:]

    print("Best error: ", best_error)
    print("Mean error: ", np.sqrt(best_error / 4))
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t

    return transformation_matrix

def compute_calibration(center, left, right):
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
    print("pcd_center: ", pcd_center.points)
    print("pcd_center shape: ", np.asarray(pcd_center.points).shape)
    pcd_left = left.get_pcd_from_rgb_depth(left.last_rgb, mean_left)
    pcd_right = right.get_pcd_from_rgb_depth(right.last_rgb, mean_right)

    original_center = copy.deepcopy(pcd_center)
    original_left = copy.deepcopy(pcd_left)
    original_right = copy.deepcopy(pcd_right)

    # Flip it, otherwise the pointcloud will be upside down
    flip_180(pcd_center)
    flip_180(pcd_left)
    flip_180(pcd_right)

    points = np.asarray(pcd_center.points)
    a_c, b_c, c_c, d_c = fit_plane_to_points(points)
    print("a_c: ", a_c, " b_c: ", b_c, " c_c: ", c_c, " d_c: ", d_c)
    plane_c = (a_c, b_c, c_c, d_c)
    points = np.asarray(pcd_left.points)
    a_l, b_l, c_l, d_l = fit_plane_to_points(points)
    plane_l = (a_l, b_l, c_l, d_l)
    points = np.asarray(pcd_right.points)
    a_r, b_r, c_r, d_r = fit_plane_to_points(points)
    plane_r = (a_r, b_r, c_r, d_r)

    corners_center = np.zeros((4,3))
    corners_left = np.zeros((4,3))
    corners_right = np.zeros((4,3))

    for i, corner in enumerate(center.quadrilateral):
        index = rgb_point_to_pcd_index(corner, center.last_rgb, np.asarray(mean_center), center, original_center)
        print(np.asarray(pcd_center.points)[index])
        corners_center[i] = np.asarray(pcd_center.points)[index][0]
        corners_center[i] = project_point_on_plane(corners_center[i], plane_c)
    for i, corner in enumerate(left.quadrilateral):
        index = rgb_point_to_pcd_index(corner, left.last_rgb, np.asarray(mean_left), left, original_left)
        corners_left[i] = np.asarray(pcd_left.points)[index][0]
        corners_left[i] = project_point_on_plane(corners_left[i], plane_l)
    for i, corner in enumerate(right.quadrilateral):
        index = rgb_point_to_pcd_index(corner, right.last_rgb, np.asarray(mean_right), right, original_right)
        corners_right[i] = np.asarray(pcd_right.points)[index][0]
        corners_right[i] = project_point_on_plane(corners_right[i], plane_r)

    meshes_c = []
    meshes = []
    for i, corner in enumerate(corners_center):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=200)
        mesh.paint_uniform_color([i*0.3, 0, 1-i*0.3])
        mesh.translate(corner, relative=False)
        meshes_c.append(mesh)
        meshes.append(mesh)
    #o3d.visualization.draw_geometries([pcd_center, *meshes_c])
    meshes_l = []
    for i, corner in enumerate(corners_left):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=200)
        mesh.paint_uniform_color([i*0.3, 0, 1-i*0.3])
        mesh.translate(corner, relative=False)
        meshes_l.append(mesh)
        meshes.append(mesh)
    #o3d.visualization.draw_geometries([pcd_left, *meshes_l])
    meshes_r = []
    for i, corner in enumerate(corners_right):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=200)
        mesh.paint_uniform_color([i*0.3, 0, 1-i*0.3])
        mesh.translate(corner, relative=False)
        meshes_r.append(meshes_r)
        meshes.append(mesh)
    o3d.visualization.draw_geometries([pcd_right, pcd_center, pcd_left, *meshes])   

    # Calcola la matrice di rotazione tra i piani
    T_l = find_rotation_translation_ransac(corners_left.T, corners_center.T)
    T_r = find_rotation_translation_ransac(corners_right.T, corners_center.T)
    print("T_l: ", T_l)
    print("T_r: ", T_r)
    pcd_left.transform(T_l)
    pcd_right.transform(T_r)
    o3d.visualization.draw_geometries([pcd_left, pcd_right, pcd_center])
    # Salvare le matrici di calibrazione in un file
    datestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if not os.path.exists("./calibrations/" + datestring):
        os.mkdir("./calibrations/" + datestring)
    np.save("./calibrations/" + datestring + "/T_l.npy", T_l)
    np.save("./calibrations/" + datestring + "/T_r.npy", T_r)


def filter_pcd(pcd):
    cl, indx = pcd.remove_radius_outlier(nb_points=20, radius=0.001)
    pcd = pcd.select_by_index(indx)
    return pcd

def acquire_shot(center, left, right, calibration_dir):
    # Acquisizione dei frame

    pcd_center, (rgb_center, depth_center) = center.get_pcd_and_frames()
    pcd_left, (rgb_left, depth_left) = left.get_pcd_and_frames()
    pcd_right, (rgb_right, depth_right) = right.get_pcd_and_frames()

    # Flip it, otherwise the pointcloud will be upside down
    flip_180(pcd_center)
    flip_180(pcd_left)
    flip_180(pcd_right)

    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right])

    # Carica le matrici
    T_l = np.load("./calibrations/" + calibration_dir + "/T_l.npy")
    T_r = np.load("./calibrations/" + calibration_dir + "/T_r.npy")

    # Applica le matrici
    points_left = np.asarray(pcd_left.points)
    points_homogeneous_l = np.hstack([points_left, np.ones((points_left.shape[0], 1))])
    transformed_points_l = np.dot(T_l, points_homogeneous_l.T).T
    pcd_left.points = o3d.utility.Vector3dVector(transformed_points_l[:, :3])

    points_right = np.asarray(pcd_right.points)
    points_homogeneous_r = np.hstack([points_right, np.ones((points_right.shape[0], 1))])
    transformed_points_r = np.dot(T_r, points_homogeneous_r.T).T
    pcd_right.points = o3d.utility.Vector3dVector(transformed_points_r[:, :3])

    #o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right])
    # Salva le pointcloud
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if not os.path.exists("./acquisitions/" + datetime_string):
        os.mkdir("./acquisitions/" + datetime_string)
    
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/center.ply", pcd_center)
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/left.ply", pcd_left)
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/right.ply", pcd_right)
    
    entire_pcd = pcd_center + pcd_left + pcd_right
    filter_pcd(entire_pcd)
    o3d.visualization.draw_geometries([entire_pcd])
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/entire.ply", entire_pcd)


calibrate_text = """"Assicurarsi di posizionare l'oggetto di calibrazione in modo che sia visibile da tutte le camere.\n
Quando si è pronti, premere il tasto "Calibra" per iniziare la procedura di calibrazione.\n"""

acquire_text = """"Assicurarsi che il soggetto sia visibile da tutte le camere.\n"""

