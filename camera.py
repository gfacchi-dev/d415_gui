import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
from utilsjacopo import flip_180, rotate_90

class Camera:
    
    def __init__(self, name, id, scala, clampMin, clampMax, rotate_image: bool = False):
        self.name = name
        self.colorizer = rs.colorizer()
        SCALA = scala
        self.scala = SCALA
        self.pipeline = rs.pipeline()
        self.rotate_image = rotate_image
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
        self.depth_queue = np.zeros((720, 1280, 90))
        self.last_frame = 0
        self.last_rgb = None
        self.quadrilateral = None

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

    def get_frame_and_add_to_queue(self):
        frame = self.get_depth_frame()
        if self.last_frame == 89:
            self.last_rgb = self.get_rgb_frame()
        else:
            self.depth_queue[:, :, self.last_frame] = frame
            self.last_frame += 1 

    def get_depth_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        depth_array = np.asanyarray(np.uint16(depth_frame.get_data()))
        if self.rotate_image:
            depth_array = np.flip(depth_array, axis=(0, 1))
        return depth_array
    
    def get_rgb_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        color_array = np.asanyarray(color_frame.get_data())
        if self.rotate_image:
            color_array = np.asanyarray(cv2.rotate(color_array, cv2.ROTATE_180))
        return color_array
    
    def get_rgb_depth(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_array = np.asanyarray(np.uint16(depth_frame.get_data()))
        color_array = np.asanyarray(color_frame.get_data())
        if self.rotate_image:
            color_array = np.asanyarray(cv2.rotate(color_array, cv2.ROTATE_180))
            depth_array = np.flip(depth_array, axis=(0, 1))
        return color_array, depth_array
    
    def get_pcd_and_frames(self, filter_pcd: bool = False):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        #depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        depth_array = np.asanyarray(np.uint16(depth_frame.get_data()))
        color_array = np.asanyarray(color_frame.get_data())
        if self.rotate_image:
            color_array = np.asanyarray(cv2.rotate(color_array, cv2.ROTATE_180))
            depth_array = np.flip(depth_array, axis=(0, 1))

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
        cv2.imwrite(f"tmpTmpColor.png", np.asanyarray(rgb))
        cv2.imwrite(f"tmpTmpDepth.png", np.uint16(depth))
        rgb = o3d.io.read_image("tmpTmpColor.png")
        depth = o3d.io.read_image("tmpTmpDepth.png")
        # Create rgbd image from color and depth images
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, convert_rgb_to_intensity=False
        )
        camera_intrinsic = self.get_intrinsics()
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
        rotate_90(pcd)
        flip_180(pcd)
        return pcd
    
    def detect_quadrilaterals(self):
        image = self.get_rgb_frame()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Trova i contorni nell'immagine con una funzione di rilevamento dei contorni
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        approxs = []
        # Trova il quadrato tra i contorni trovati
        for contour in contours:
            # Approssima il contorno a una forma più semplice (un poligono)
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Se il poligono ha 4 lati, potrebbe essere un quadrato
            if len(approx) == 4:
                # Calcola l'area del quadrato
                area = cv2.contourArea(approx)
                if area > 0:
                    x, y, w, h = cv2.boundingRect(approx)
                    center_square = gray[y:y+h, x:x+w]
                    _, thresholded = cv2.threshold(center_square, 100, 255, cv2.THRESH_BINARY)
                    percentage_black = np.count_nonzero(thresholded == 0) / (w * h) * 100
                    if percentage_black > 70:
                        approxs.append((approx, area, percentage_black))

        # Sort the approxs by area
        approxs.sort(key=lambda x: (x[1],x[2]), reverse=True)
        if len(approxs) > 0:
            square = np.squeeze(approxs[0][0][:])
            center = np.mean(square, axis=0)
            diff = square - center
            points = np.zeros((4, 2), dtype=np.int32)
            for indx, element in enumerate(np.asarray(square)):
                if diff[indx][0] < 0 and diff[indx][1] < 0:
                    points[0] = element
                elif diff[indx][0] > 0 and diff[indx][1] < 0:
                    points[1] = element
                elif diff[indx][0] < 0 and diff[indx][1] > 0:
                    points[2] = element
                elif diff[indx][0] > 0 and diff[indx][1] > 0:
                    points[3] = element
            # Il punto con x minore e y maggiore
            self.last_rgb = image
            self.quadrilateral = points
            cv2.drawContours(image, [square], 0, (0, 255, 0), 2)  # Disegna il contorno del quadrato
            # Disegna 4 cerchi ai vertici del quadrato di colori diversi
            cv2.circle(image, tuple(self.quadrilateral[0]), 5, (255,0,0), -1)
            cv2.circle(image, tuple(self.quadrilateral[1]), 5, (0,255,0), -1)
            cv2.circle(image, tuple(self.quadrilateral[2]), 5, (0,0,255), -1)
            cv2.circle(image, tuple(self.quadrilateral[3]), 5, (0,0,0), -1)

        else:
            square = None
        return image, square