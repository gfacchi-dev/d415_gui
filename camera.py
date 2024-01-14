import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2

class Camera:
    
    def __init__(self, name, id, scala, clampMin, clampMax):
        self.name = name
        self.colorizer = rs.colorizer()
        SCALA = scala
        self.scala = SCALA
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(id)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        advnc_mode = rs.rs400_advanced_mode(device)
        current_std_depth_table = advnc_mode.get_depth_table()
        current_std_depth_table.depthClampMin = clampMin
        current_std_depth_table.depthClampMax = clampMax*SCALA
        current_std_depth_table.depthUnits = int(1000/SCALA)  # mm

        advnc_mode.set_depth_table(current_std_depth_table)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.depth)

        # Start cameras pipelines using configuration provided earlier
        cfg = self.pipeline.start(config)
        profile = cfg.get_stream(rs.stream.depth)
        self.intr = profile.as_video_stream_profile().get_intrinsics()

    def get_intrinsics(self):
        intr = self.intr
        return o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(
            intr.width,
            intr.height,
            intr.fx,
            intr.fy,
            intr.ppx,
            intr.ppy,
        )
    )

    def get_pcd_and_frames(self, filter_pcd: bool = False):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        #depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        depth_array = np.asanyarray(np.uint16(depth_frame.get_data()))
        color_array = np.asanyarray(color_frame.get_data())

        cv2.imwrite(f"tmpDepth.png", np.uint16(depth_array))
        cv2.imwrite(f"tmpColor.png", np.asanyarray(color_array))
        rgb = o3d.io.read_image("tmpColor.png")
        depth = o3d.io.read_image("tmpDepth.png")
        # Create rgbd image from color and depth images

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, convert_rgb_to_intensity=False
        )
        camera_intrinsic = self.get_intrinsics()
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
        if filter_pcd:
            cl, indx = pcd.remove_radius_outlier(nb_points=100, radius=0.02)
            pcd = pcd.select_by_index(indx)

        return pcd, (color_array, depth_array)
    
    def get_pcd_from_rgb_depth(self, rgb, depth):
        cv2.imwrite(f"tmpTmpDepth.png", np.uint16(depth))
        cv2.imwrite(f"tmpTmpColor.png", np.asanyarray(rgb))
        rgb = o3d.io.read_image("tmpTmpColor.png")
        depth = o3d.io.read_image("tmpTmpDepth.png")
        # Create rgbd image from color and depth images
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, convert_rgb_to_intensity=False
        )
        camera_intrinsic = self.get_intrinsics()
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
        return pcd