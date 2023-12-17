import open3d as o3d
import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import cv2 

SCALA = 3

pipeline = rs.pipeline()
config = rs.config()
config.enable_device("210622061176")
colorizer = rs.colorizer()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
advnc_mode = rs.rs400_advanced_mode(device)
current_std_depth_table = advnc_mode.get_depth_table()
current_std_depth_table.depthClampMin = 100
current_std_depth_table.depthClampMax = 800*SCALA
current_std_depth_table.depthUnits = int(1000/SCALA)  # mm

advnc_mode.set_depth_table(current_std_depth_table)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
align = rs.align(rs.stream.depth)

# Start cameras pipelines using configuration provided earlier
cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.depth)
intr_center = profile.as_video_stream_profile().get_intrinsics()

frames = pipeline.wait_for_frames()
frames = align.process(frames)
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()
depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
depth_array = np.asanyarray(np.uint16(depth_frame.get_data()))
color_array = np.asanyarray(color_frame.get_data())
cv2.imwrite("temp/COLORprovaaaaaa.png", np.asanyarray(color_frame.get_data()))

prova = cv2.imread("temp/COLORprovaaaaaa.png")
gray_image = cv2.cvtColor(prova, cv2.COLOR_BGR2GRAY)

# Applica una soglia per ottenere solo i pixel neri (bianco diventa nero)
_, black_and_white = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inverted_image = 255-black_and_white

masked_rgb = cv2.bitwise_and(color_array, color_array, mask=inverted_image)
masked_depth = cv2.bitwise_and(depth_array, depth_array, mask=inverted_image)


# Visualizza l'immagine risultante
cv2.imshow('Masked', masked_rgb)
cv2.waitKey(0)
cv2.imshow('Masked', masked_depth)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("temp/COLORprovaaaaaa.png", np.asanyarray(masked_rgb))
cv2.imwrite("temp/provaaaaaa.png", np.uint16(np.asanyarray(masked_depth)))


depth_raw_center = o3d.io.read_image("temp/provaaaaaa.png")
color_raw_center = o3d.io.read_image("temp/COLORprovaaaaaa.png")
rgbd_image_center = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw_center, depth_raw_center, convert_rgb_to_intensity=False
    )
camera_intrinsic_center = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(
            intr_center.width,
            intr_center.height,
            intr_center.fx,
            intr_center.fy,
            intr_center.ppx,
            intr_center.ppy,
        )
    )

pcd_center = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_center, camera_intrinsic_center
    )

o3d.visualization.draw_geometries([pcd_center])