import cv2
from utils import write_filtered_image
import numpy as np
import open3d as o3d

def test():
    color_image = cv2.imread("RGB_SQUARES.png")
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,150,100])
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
    res = cv2.bitwise_and(color_image, color_image, mask=mask)
    cv2.imshow("prova", res)
    cv2.waitKey(0)
    return

def boh():
    color_image = cv2.imread("./temp/centerColor.png")
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) 
    print(color_image.shape)
    depth_image = cv2.imread("./temp/centerDepth.png")
    print(depth_image.shape)
    write_filtered_image(color_image, depth_image, "TESTTEST")
    depth_raw = o3d.io.read_image("temp/TESTTESTDepth.png")
    color_raw = o3d.io.read_image("temp/TESTTESTColor.png")

    rgbd_image_left = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, convert_rgb_to_intensity=False
        )

test()