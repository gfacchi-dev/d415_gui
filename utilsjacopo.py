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
    # medians[i][j] = sorted_tensor[median_indexes[i][j]]
    medians = torch.gather(tensor, 2, median_indexes.unsqueeze(2).repeat(1,1,90))
    # Calculate mean ignoring zero values
    # mean_values = torch.where(non_zero_mask, tensor, torch.zeros_like(tensor))
    # sum_mean_values = mean_values.sum(dim=2)
    # mean_values = torch.where(
    #     num_non_zero_elements.squeeze(2) > 0,
    #     sum_mean_values / num_non_zero_elements.sum(dim=2),
    #     torch.zeros_like(sum_mean_values),
    # )

    sum_values = tensor.sum(dim=2)
    
    mean_values = sum_values / num_non_zero_elements

    # Calculate variance ignoring zero values
    # diff_squared = torch.where(
    #     non_zero_mask,
    #     (tensor - mean_values.unsqueeze(2)) ** 2,
    #     torch.zeros_like(tensor),
    # )
    # variance_values = diff_squared.sum(dim=2)

    diff_squared = torch.where(
        non_zero_mask, 
        (tensor - mean_values.unsqueeze(2).repeat(1,1,90)) ** 2,
        torch.zeros_like(tensor)
    )
    variance_values = diff_squared.sum(dim=2)
    num_non_zero_elements_v = num_non_zero_elements
    variance_values = torch.div(variance_values, num_non_zero_elements_v)
    # variance_values = torch.where(
    #     variance_values == torch.nan,
    #     torch.sub(torch.zeros_like(num_non_zero_elements), 1),
    #     variance_values
    # )
    variance_values = torch.where(
        num_non_zero_elements != 0,
        variance_values,
        torch.sub(torch.zeros_like(num_non_zero_elements), 1)
    )
    # Set -1 where num_non_zero_elements is 0
    # variance_values[num_non_zero_elements.squeeze(2) == 0] = -1

    return mean_values, variance_values, num_non_zero_elements, medians

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


def fit_func(params, A, B):
    R = params[:9].reshape(3, 3)
    t = params[9:]

    transformed_A = np.dot(R, A) + t[:, np.newaxis]
    error = B - transformed_A
    return np.sum(error**2)

def find_rotation_translation_ransac(A, B, num_iterations=1000, tolerance=1e-5):
    best_params = None
    best_error = float('inf')

    for _ in range(num_iterations):
        # Seleziona un set casuale di punti
        indices = np.random.choice(A.shape[1], size=4, replace=False)
        sampled_A = A[:, indices]
        sampled_B = B[:, indices]

        # Utilizza i minimi quadrati per trovare la rototraslazione
        result = minimize(fit_func, x0=np.zeros(12), args=(sampled_A, sampled_B), method='Powell')

        # Calcola l'errore complessivo
        total_error = fit_func(result.x, A, B)

        # Aggiorna i migliori parametri se necessario
        if total_error < best_error:
            best_error = total_error
            best_params = result.x

        # Termina se l'errore è inferiore alla tolleranza
        if best_error < tolerance:
            break

    # Estrai la matrice di trasformazione dai migliori parametri
    R = best_params[:9].reshape(3, 3)
    t = best_params[9:]

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
    # Rotate of 90 degrees around z axis
    rotate_90(pcd_center, clockwise=True)
    rotate_90(pcd_left, clockwise=True)
    rotate_90(pcd_right, clockwise=True)

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
    o3d.visualization.draw_geometries([pcd_center, *meshes_c])
    meshes_l = []
    for i, corner in enumerate(corners_left):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=200)
        mesh.paint_uniform_color([i*0.3, 0, 1-i*0.3])
        mesh.translate(corner, relative=False)
        meshes_l.append(mesh)
        meshes.append(mesh)
    o3d.visualization.draw_geometries([pcd_left, *meshes_l])
    meshes_r = []
    for i, corner in enumerate(corners_right):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=200)
        mesh.paint_uniform_color([i*0.3, 0, 1-i*0.3])
        mesh.translate(corner, relative=False)
        meshes_r.append(meshes_r)
        meshes.append(mesh)
    o3d.visualization.draw_geometries([pcd_right, pcd_left, pcd_center, *meshes])   

    # Calcola la matrice di rotazione tra i piani
    T_l = find_rotation_translation_ransac(corners_left.T, corners_center.T)
    T_r = find_rotation_translation_ransac(corners_right.T, corners_center.T)
    print("T_l: ", T_l)
    print("T_r: ", T_r)
    pcd_left.transform(T_l)
    pcd_right.transform(T_r)
    o3d.visualization.draw_geometries([pcd_left, pcd_center, pcd_right])
    # Salvare le matrici di calibrazione in un file
    datestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if not os.path.exists("./calibrations/" + datestring):
        os.mkdir("./calibrations/" + datestring)
    np.save("./calibrations/" + datestring + "/T_l.npy", T_l)
    np.save("./calibrations/" + datestring + "/T_r.npy", T_r)




def old_compute_calibration(center, left, right):
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
    rotate_90(pcd_center, clockwise=True)
    rotate_90(pcd_left, clockwise=True)
    rotate_90(pcd_right, clockwise=True)

    # Show pcd
    #o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd_center.points)
    a_c, b_c, c_c, d_c = fit_plane_to_points(points)
    plane_c = (a_c, b_c, c_c, d_c)
    points = np.asarray(pcd_left.points)
    a_l, b_l, c_l, d_l = fit_plane_to_points(points)
    plane_l = (a_l, b_l, c_l, d_l)
    points = np.asarray(pcd_right.points)
    a_r, b_r, c_r, d_r = fit_plane_to_points(points)
    plane_r = (a_r, b_r, c_r, d_r)
    # REFERENCE PLANE
    p_ref = [0., 0., 1., 0.]

    # Draw plane 
    center_plane = get_plane_pcd(plane_c, color=[1, 0, 0])
    left_plane = get_plane_pcd(plane_l, color=[0, 1, 0])
    right_plane = get_plane_pcd(plane_r, color=[0, 0, 1])
    ref_plane = get_plane_pcd(p_ref, color=[0, 0, 0])

    #o3d.visualization.draw_geometries([pcd_left, left_plane, pcd_right, right_plane, pcd_center, center_plane])

    # Calcola la matrice di rotazione tra i piani
    R_left = compute_rotation(plane_l, p_ref)
    R_right = compute_rotation(plane_r, p_ref)
    R_center = compute_rotation(plane_c, p_ref)

    pcd_left.rotate(R_left, center=(0,0,0))
    left_plane.rotate(R_left, center=(0,0,0))
    #o3d.visualization.draw_geometries([pcd_left, left_plane, ref_plane])
    
    #t_a, t_b, t_c, d = fit_plane_to_points(np.asarray(pcd_left.points))
    _, _, _, d = fit_plane_to_points(np.asarray(left_plane.points))
    _, _, _, d2 = fit_plane_to_points(np.asarray(pcd_left.points))
    # print("d_pcd: ", d, " d_plane: ", d2)
    t_left = np.array([0,0,-d2], dtype=np.float64)
    pcd_left.translate(t_left, relative=True)
    left_plane.translate(t_left, relative=True)
    #o3d.visualization.draw_geometries([pcd_left, left_plane, ref_plane])

    pcd_right.rotate(R_right, center=(0,0,0))
    right_plane.rotate(R_right, center=(0,0,0))
    _, _, _, d = fit_plane_to_points(np.asarray(right_plane.points))
    _, _, _, d2 = fit_plane_to_points(np.asarray(pcd_right.points))
    print("d_pcd: ", d2, " d_plane: ", d)
    t_right = np.array([0,0,-d2], dtype=np.float64)
    pcd_right.translate(t_right, relative=True)
    right_plane.translate(t_right, relative=True)
    #o3d.visualization.draw_geometries([pcd_right, right_plane, ref_plane])

    pcd_center.rotate(R_center)
    center_plane.rotate(R_center)
    _, _, _, d = fit_plane_to_points(np.asarray(center_plane.points))
    _, _, _, d2 = fit_plane_to_points(np.asarray(pcd_center.points))
    print("d_pcd: ", d2, " d_plane: ", d)
    t_center = np.array([0,0,-d2], dtype=np.float64)
    pcd_center.translate(t_center, relative=True)
    center_plane.translate(t_center, relative=True)
    #o3d.visualization.draw_geometries([pcd_center, center_plane, ref_plane])

    #o3d.visualization.draw_geometries([left_plane, center_plane, right_plane])
    o3d.visualization.draw_geometries([pcd_left, pcd_center, pcd_right])

    image_c = center.last_rgb
    image_l = left.last_rgb
    image_r = right.last_rgb

    corner_center_1 = center.quadrilateral[0]
    print("corner_center_1: ", corner_center_1)
    print("corner_center_1 shape: ", corner_center_1.shape)
    print("corner_center_1: ", corner_center_1)
    corner_center_2 = center.quadrilateral[1]
    print("corner_center_2: ", corner_center_2)

    corner_left_1 = left.quadrilateral[0]
    print("corner_left_1: ", corner_left_1)
    corner_left_2 = left.quadrilateral[1]
    print("corner_left_2: ", corner_left_2)

    corner_right_1 = right.quadrilateral[0]
    print("corner_right_1: ", corner_right_1)
    corner_right_2 = right.quadrilateral[1]
    print("corner_right_2: ", corner_right_2)


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
    rot_center_left = np.asarray(pcd_left.points)[corner_l1_index][0]
    pcd_left.rotate(R_L2C, center=rot_center_left)
    
    o3d.visualization.draw_geometries([pcd_center, pcd_left])
    v1 = corner_r_1 - corner_r_2
    v2 = corner_c_1 - corner_c_2
    R_R2C = matrix_between_vectors(v2, v1, clockwise=False)
    rot_center_right = np.asarray(pcd_right.points)[corner_r1_index][0]
    pcd_right.rotate(R_R2C, center=rot_center_right)

    o3d.visualization.draw_geometries([pcd_center, pcd_right])

    print("R_L2C: ", R_L2C)
    print("T_L2C: ", T_L2C)
    print("R_R2C: ", R_R2C)
    print("T_R2C: ", T_R2C)
    # Sistema di riferimento di open3d
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])


    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right, mesh_frame])

    # Salvare le matrici di calibrazione in un file
    datestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if not os.path.exists("./calibrations/" + datestring):
        os.mkdir("./calibrations/" + datestring)
    np.save("./calibrations/" + datestring + "/R_l1.npy", R_left)
    np.save("./calibrations/" + datestring + "/R_r1.npy", R_right)
    np.save("./calibrations/" + datestring + "/R_c1.npy", R_center)
    np.save("./calibrations/" + datestring + "/t_l1.npy", t_left)
    np.save("./calibrations/" + datestring + "/t_r1.npy", t_right)
    np.save("./calibrations/" + datestring + "/t_c1.npy", t_center)
    np.save("./calibrations/" + datestring + "/R_l2.npy", R_L2C)
    np.save("./calibrations/" + datestring + "/R_r2.npy", R_R2C)
    np.save("./calibrations/" + datestring + "/t_l2.npy", T_L2C)
    np.save("./calibrations/" + datestring + "/t_r2.npy", T_R2C)
    np.save("./calibrations/" + datestring + "/rotation_center_left.npy", rot_center_left)
    np.save("./calibrations/" + datestring + "/rotation_center_right.npy", rot_center_right)

def old_acquire_shot(center, left, right, calibration_dir):
    # Acquisizione dei frame

    pcd_center, (rgb_center, depth_center) = center.get_pcd_and_frames()
    pcd_left, (rgb_left, depth_left) = left.get_pcd_and_frames()
    pcd_right, (rgb_right, depth_right) = right.get_pcd_and_frames()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    # Flip it, otherwise the pointcloud will be upside down
    flip_180(pcd_center)
    flip_180(pcd_left)
    flip_180(pcd_right)

    # Rotate of 90 degrees around z axis
    rotate_90(pcd_center, clockwise=True)
    rotate_90(pcd_left, clockwise=True)
    rotate_90(pcd_right, clockwise=True)
    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right, mesh_frame])

    # Carica le matrici
    R_l1 = np.load("./calibrations/" + calibration_dir + "/R_l1.npy")
    R_r1 = np.load("./calibrations/" + calibration_dir + "/R_r1.npy")
    R_c1 = np.load("./calibrations/" + calibration_dir + "/R_c1.npy")
    t_l1 = np.load("./calibrations/" + calibration_dir + "/t_l1.npy")
    t_r1 = np.load("./calibrations/" + calibration_dir + "/t_r1.npy")
    t_c1 = np.load("./calibrations/" + calibration_dir + "/t_c1.npy")
    R_l2 = np.load("./calibrations/" + calibration_dir + "/R_l2.npy")
    R_r2 = np.load("./calibrations/" + calibration_dir + "/R_r2.npy")
    t_l2 = np.load("./calibrations/" + calibration_dir + "/t_l2.npy")
    t_r2 = np.load("./calibrations/" + calibration_dir + "/t_r2.npy")
    rotation_center_left = np.load("./calibrations/" + calibration_dir + "/rotation_center_left.npy")
    rotation_center_right = np.load("./calibrations/" + calibration_dir + "/rotation_center_right.npy")

    # Applica le matrici
    pcd_center.rotate(R_c1, center=(0,0,0))
    pcd_left.rotate(R_l1, center=(0,0,0))
    pcd_right.rotate(R_r1, center=(0,0,0))
    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right, mesh_frame])

    pcd_center.translate(t_c1, relative=True)
    pcd_left.translate(t_l1, relative=True)
    pcd_right.translate(t_r1, relative=True)
    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right, mesh_frame])

    pcd_left.translate(t_l2, relative=True)
    pcd_right.translate(t_r2, relative=True)
    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right, mesh_frame])

    pcd_left.rotate(R_l2, center=rotation_center_left)
    pcd_right.rotate(R_r2, center=rotation_center_right)

    # Salva le pointcloud
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    if not os.path.exists("./acquisitions/" + datetime_string):
        os.mkdir("./acquisitions/" + datetime_string)
    
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/center.ply", pcd_center)
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/left.ply", pcd_left)
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/right.ply", pcd_right)


    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right, mesh_frame])
    merged_pcd = pcd_center + pcd_left + pcd_right
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/merged.ply", merged_pcd)

    o3d.visualization.draw_geometries([merged_pcd, mesh_frame])
    return



def acquire_shot(center, left, right, calibration_dir):
    # Acquisizione dei frame

    pcd_center, (rgb_center, depth_center) = center.get_pcd_and_frames()
    pcd_left, (rgb_left, depth_left) = left.get_pcd_and_frames()
    pcd_right, (rgb_right, depth_right) = right.get_pcd_and_frames()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    # Flip it, otherwise the pointcloud will be upside down
    flip_180(pcd_center)
    flip_180(pcd_left)
    flip_180(pcd_right)

    # Rotate of 90 degrees around z axis
    rotate_90(pcd_center, clockwise=True)
    rotate_90(pcd_left, clockwise=True)
    rotate_90(pcd_right, clockwise=True)
    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right, mesh_frame])

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

    o3d.visualization.draw_geometries([pcd_center, pcd_left, pcd_right, mesh_frame])
    # Salva le pointcloud
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if not os.path.exists("./acquisitions/" + datetime_string):
        os.mkdir("./acquisitions/" + datetime_string)
    
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/center.ply", pcd_center)
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/left.ply", pcd_left)
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/right.ply", pcd_right)
    
    entire_pcd = pcd_center + pcd_left + pcd_right
    o3d.io.write_point_cloud(f"./acquisitions/{datetime_string}/entire.ply", entire_pcd)



calibrate_text = """"Assicurarsi di posizionare l'oggetto di calibrazione in modo che sia visibile da tutte le camere.\n
Quando si è pronti, premere il tasto "Calibra" per iniziare la procedura di calibrazione.\n"""

acquire_text = """"Assicurarsi che il soggetto sia visibile da tutte le camere.\n"""

