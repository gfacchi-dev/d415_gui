import numpy as np
import torch
from torch.nn.functional import interpolate
from scipy.interpolate import interp1d
import open3d as o3d
import cv2

class FrameQueue:
    def __init__(self, max_frames, frame_shape):
        self.max_frames = max_frames
        self.frame_shape = frame_shape
        self.buffer = np.empty(
            (max_frames,) + frame_shape,
            dtype=np.uint8 if frame_shape[-1] == 3 else np.uint16,
        )
        self.index = 0
        self.current_size = 0

    def add_frame(self, frame):
        if frame.shape != self.frame_shape:
            raise ValueError("Frame dimensions do not match the expected shape.")

        if self.current_size < self.max_frames:
            self.current_size += 1

        self.buffer[self.index] = frame
        self.index = (self.index + 1) % self.max_frames

    def get_last_frame(self):
        if self.current_size == 0:
            return None

        last_frame = self.buffer[(self.index - 1) % self.max_frames]
        return last_frame.copy()

    def get_frames_as_tensor(self):
        if self.current_size == 0:
            return torch.empty(0, *self.frame_shape, dtype=torch.uint8)

        if self.index == 0:
            frames = self.buffer[: self.current_size]
        else:
            frames = np.concatenate(
                (self.buffer[self.index :], self.buffer[: self.index])
            )

        # Stack frames along the third dimension to form a tensor
        frames_tensor = torch.from_numpy(np.stack(frames, axis=-1, dtype=np.float32))

        return frames_tensor

    def __len__(self):
        return self.current_size


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


def get_maps(variances, means, threshold=None):
    max_value = variances.max()
    invalid_indexes = np.argwhere(variances == -1)
    valid_variances = np.copy(variances)
    valid_variances[invalid_indexes[:, 0], invalid_indexes[:, 1]] = max_value
    if threshold is None:
        threshold = np.median(variances[variances != -1])
    print(f"Threshold: {np.median(variances[variances!=-1])}")

    from scipy.interpolate import interp1d

    img_indexes = np.argwhere(valid_variances < threshold)
    high_variance_indexes = np.argwhere(valid_variances >= threshold)

    filtered_means = np.copy(means)
    filtered_means[high_variance_indexes[:, 0], high_variance_indexes[:, 1]] = 0
    zero_variance_indexes = np.argwhere(valid_variances == 0)
    zero_variance_image = np.zeros((720, 1280))
    zero_variance_image[zero_variance_indexes[:, 0], zero_variance_indexes[:, 1]] = 255

    variance_image = np.zeros((720, 1280))
    m = interp1d([valid_variances.min(), threshold], [0, 254])
    variance_image[img_indexes[:, 0], img_indexes[:, 1]] = m(
        valid_variances[img_indexes[:, 0], img_indexes[:, 1]]
    )

    variance_image[high_variance_indexes[:, 0], high_variance_indexes[:, 1]] = 255

    return variance_image, zero_variance_image, threshold, filtered_means


def save_pcl(pointcloud1, pointcloud2, pointcloud3, folder):
    p1_colors = pointcloud1.colors
    p2_colors = pointcloud2.colors
    p3_colors = pointcloud3.colors
    p1_load = pointcloud1.points
    p2_load = pointcloud2.points
    p3_load = pointcloud3.points
    
    p4_colors = np.concatenate((p1_colors, p2_colors, p3_colors), axis=0)
    p4_load = np.concatenate((p1_load, p2_load, p3_load), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p4_load)
    pcd.colors = o3d.utility.Vector3dVector(p4_colors)
    o3d.io.write_point_cloud(
        f"./acquisitions/{folder}/pcl_l.pcd",
        pointcloud1,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )
    o3d.io.write_point_cloud(
        f"./acquisitions/{folder}/pcl_r.pcd",
        pointcloud2,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )
    o3d.io.write_point_cloud(
        f"./acquisitions/{folder}/pcl_c.pcd",
        pointcloud3,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )
    o3d.io.write_point_cloud(
        f"./acquisitions/{folder}/pcl.pcd",
        pcd,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )
    return True

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
    #display_inlier_outlier(pcd_down, ind)
    pcd_down = cl

    radius_normal = voxel_size * 4
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def preprocess_point_cloud(pcd, radius_normal, radius_feature):
    print(":: NO Downsample")
    
    #cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
    #display_inlier_outlier(pcd_down, ind)

    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 1))
    return result


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


from scipy.spatial.transform import Rotation   

def get_angles_from_transform_matrix(matrix):
    x = matrix[0][0:3]
    y = matrix[1][0:3]
    z = matrix[2][0:3]
    rotation_matrix = np.array([matrix[0][0:3],matrix[1][0:3],matrix[2][0:3]])
    t = [matrix[0][3], matrix[1][3], matrix[2][3]]
    r =  Rotation.from_matrix(rotation_matrix)
    angles = r.as_euler("xyz",degrees=True)
    return angles[0], angles[1], angles[2]

def write_filtered_image(color_image, depth_image, name):
    # Remove white
    # rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # gray_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # black_and_white = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)
    # inverted_image = 255-black_and_white
    # rgb = cv2.bitwise_and(rgb, rgb, mask=inverted_image)

    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,100,50])
    upper_blue = np.array([130,255,255])
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    lower_red = np.array([0,150,100])
    upper_red = np.array([5,255,255])
    red_mask_1 = cv2.inRange(hsv_image, lower_red, upper_red)
    lower_red = np.array([175,150,100])
    upper_red = np.array([180,255,255])
    red_mask_2 = cv2.inRange(hsv_image, lower_red, upper_red)
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    lower_green = np.array([40,150,100])
    upper_green = np.array([70,255,255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    #Filter only the red colour from the original image using the mask(foreground)
    mask = cv2.bitwise_or(blue_mask, red_mask)
    mask = cv2.bitwise_or(mask, green_mask)
    color_image_filtered = cv2.bitwise_and(color_image, color_image, mask=mask)
    depth_image_filtered = cv2.bitwise_and(depth_image, depth_image, mask=mask)
    cv2.imshow("prova", color_image_filtered)
    cv2.waitKey(0)

    cv2.imwrite(f"temp/{name}Depth.png", np.uint16(depth_image_filtered))
    cv2.imwrite(f"temp/{name}Color.png", np.asanyarray(color_image_filtered))