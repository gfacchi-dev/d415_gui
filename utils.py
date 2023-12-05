import numpy as np
import torch
from torch.nn.functional import interpolate
from scipy.interpolate import interp1d
import open3d as o3d


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
    print(variance_values)
    print(torch.min(variance_values))
    print(torch.max(variance_values))
    # Set -1 where num_non_zero_elements is 0
    # variance_values[num_non_zero_elements.squeeze(2) == 0] = -1

    return mean_values, variance_values, num_non_zero_elements


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


def save_pcl(pointcloud1, pointcloud2, folder):
    p1_colors = pointcloud1.colors
    p2_colors = pointcloud2.colors
    p1_load = pointcloud1.points
    p2_load = pointcloud2.points
    p3_colors = np.concatenate((p1_colors, p2_colors), axis=0)
    p3_load = np.concatenate((p1_load, p2_load), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p3_load)
    pcd.colors = o3d.utility.Vector3dVector(p3_colors)
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
        f"./acquisitions/{folder}/pcl.pcd",
        pcd,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )
    return True
