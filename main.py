from tkinter import *
import ttkbootstrap as tb
import pyautogui as pg
import cv2
import PIL.Image
import PIL.ImageTk
import pyrealsense2 as rs
import numpy as np
import pickle
import open3d as o3d
import math
import itertools
import threading as tr
import os
import datetime
import time
from scipy.interpolate import interp1d

from utils import ( 
    mean_variance_and_inliers_along_third_dimension_ignore_zeros,
    get_maps,
    save_pcl,
)

from utils2 import (  
    create_depth_queue,
    create_rgb_queue,
    is_far_from_camera,
    is_frontal_face,
    is_neutral_face,
    verify_calibration,
    update_gui,
    set_none,
    log_status,
    initialize,
    crop_into_the_center,
    zoom_in
)

from gui import (  
    menu_configuration,
    set_default_labels,
    buttons_trigger,
    buttons_shut,
    meter_update,
    fullscreen
)



# Modeling the GUI
#Create window
main_window = tb.Window(themename="superhero")
main_window.title("Realsense")


#Resolution
native_width, native_height = pg.size()
res = f"{native_width}x{native_height}"
main_window.geometry(res)


#Icon:
icon_path = "./images/"
icon = tb.PhotoImage(file= icon_path + "logo.png")
main_window.iconphoto(True,icon)


#Set the title lable
title_lable = tb.Label(master=main_window, text="Realsense",  font=("Arial", 30), foreground="white")
title_lable.place(x=30, y=40, anchor="nw")


#Divisor
separator = tb.Separator(orient="horizontal", master= main_window)
separator.place(x=30, y=100, anchor="nw", width= native_width * (30/100))


#Bind commands
main_window.bind("<Escape>", lambda _: main_window.quit())


#Container for buttons
button_frame = tb.Labelframe(master=main_window, text="Buttons",height=int(native_height * (20/100)), width= int(native_width * (35/100)))
button_frame.place(x=30, y=115, anchor="nw", width= native_width * (30/100))


#Button style set up        
calibrate_button_style = tb.Style()        
calibrate_button_style.configure("warning.Outline.TButton",font=("Arial", 20))

acquire_button_style = tb.Style()        
acquire_button_style.configure("info.Outline.TButton",font=("Arial", 20))

build_button_style = tb.Style()        
build_button_style.configure("primary.Outline.TButton",font=("Arial", 20))


# New buttons
calibrate_button = tb.Button(button_frame, text="CALIBRATE", state="enabled", width=20, style="warning.Outline.TButton",bootstyle = "warning-outline" ,command=lambda: calibrate())
acquire_button = tb.Button(button_frame, text="ACQUIRE", state="disabled",  width=20, style="info.Outline.TButton",bootstyle = "info-outline" ,command=lambda: show_confirmation_window())
build_button = tb.Button(button_frame, text="BUILD",  state="disabled",  width=20,  style="primary.Outline.TButton",bootstyle = "success-outline" ,command=lambda: show_confirmation_window())


calibrate_button.place(x=110, y=20)
acquire_button.place(x=110, y=80)  
build_button.place(x=110, y = 140)  


#Container for labels
label_frame = tb.Labelframe(master=main_window, text="Detections",height = int(native_height * (20/100)), width= int(native_width * (35/100)))
label_frame.place(x=30, y=380, anchor="nw", width= native_width * (30/100), height= native_height * (55/100))

#Container for realtime 
real_time_frame = tb.LabelFrame(master=main_window, text="Realtime",height = int(native_height * (40/100)), width= int(native_width * (35/100)))
real_time_frame.place(x= native_width - 650, y=native_height // 2.2, anchor="center", width= native_width * (65/100), height= native_height * (90/100))

#Button to start or stop detections
detection_button_style = tb.Style()        
detection_button_style.configure("success.Outline.TButton",font=("Arial", 12))
detection_button = tb.Button(label_frame, text="RUN DETECTION",  state="enabled",  width=20,  style="success.Outline.TButton",bootstyle = "success-outline" ,command=lambda: go_open_camera())


stop_detection_button_style = tb.Style()        
stop_detection_button_style.configure("danger.Outline.TButton",font=("Arial", 12))
stop_detection_button = tb.Button(label_frame, text="STOP DETECTION",  state="disabled", width=20,  style="danger.Outline.TButton",bootstyle = "danger-outline" ,command=lambda: stop_open_camera())


#Labels
label_frontal_face = tb.Label(text="Checking...", master=label_frame, font=("Helvetica", 15), width = 20, bootstyle = "secondary")
label_distance_face = tb.Label(text="Checking...", master=label_frame, font=("Helvetica", 15), width = 15, bootstyle = "secondary")
label_neutral_face = tb.Label(text="Checking...", master=label_frame, font=("Helvetica", 15), width = 20, bootstyle = "secondary")
label_found_face = tb.Label(text="Checking...", master=label_frame, font=("Helvetica", 15), width = 15, bootstyle = "secondary")

stop_detection_button.place(y = 20, x = 300)
detection_button.place(y = 20, x = 60)
label_distance_face.place(x=340, y=180)
label_found_face.place(x=340, y=80)
label_neutral_face.place(x=50, y=80)
label_frontal_face.place(x=50, y=180)


#Meter
meter = tb.Meter(master= label_frame, metersize=350, amountused=0, bootstyle="success", textright="%",  metertype="semi", subtext="Analysing...", subtextfont="Arial", amounttotal=100, stepsize=25)
meter.place(relx=0.5, rely=1.0, anchor="s")



#Directories analysis
directory_path = "./calibrations/"
verify_calibration(directory_path)



# Binds
def bind_event_handlers(window):
    window.bind("<A>", lambda _: show_confirmation_window())
    window.bind("<a>", lambda _: show_confirmation_window())
    window.bind("<B>", lambda _: show_confirmation_window())
    window.bind("<b>", lambda _: show_confirmation_window())

    
    
def unbind_event_handlers(window):
    window.unbind("<A>")
    window.unbind("<a>")
    window.unbind("<b>")
    window.unbind("<B>")

#Finishing modeling gui 













# PYREALSENSE
# Start configuration for the Camera Right and Camera Left
# Camera Right
# Created a pipeline object managing and processing RealSense Data Stream for computer vision
pipeline = rs.pipeline()

# Created a configuration object
config = rs.config()

# Use the camera corresponding to his serial number
config.enable_device("032622070359")

# Use a wrapper for the papeline and resolve configuration settings.
# A wrapper object provides additional capabilities and methods for interacting with the RealSense pipeline
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)

# Getter (device) from the papeline
device = pipeline_profile.get_device()

# Activate advance mode for enhanced controls
advnc_mode = rs.rs400_advanced_mode(device)

# Get depth table with minim and max depth values (in order to obtain the depth matrix)
current_std_depth_table = advnc_mode.get_depth_table()
current_std_depth_table.depthClampMin = 250
current_std_depth_table.depthClampMax = 800

# Set Depth unit
current_std_depth_table.depthUnits = 1000  # mm
print(current_std_depth_table)

# Set depth camera with previous depthclamp
advnc_mode.set_depth_table(current_std_depth_table)

# Enable stream depth and color, in 1280x720 res, format, and fps - I required my algoritm's performance to be superior 30fps
# in order not to drop essentials frames
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Align depth frame with color frame -
# synchronized the depth frame and color frame making easy to associate depth informations to its relative RGB color informations
# In oder to have that each depth frames corresponding to the same color informations
align = rs.align(rs.stream.depth)



# Camera Left
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device("233722072412")
pipeline_wrapper_2 = rs.pipeline_wrapper(pipeline_2)
pipeline_profile_2 = config_2.resolve(pipeline_wrapper_2)
device_2 = pipeline_profile_2.get_device()
advnc_mode_2 = rs.rs400_advanced_mode(device_2)
current_std_depth_table_2 = advnc_mode_2.get_depth_table()
current_std_depth_table_2.depthClampMin = 250
current_std_depth_table_2.depthClampMax = 800
current_std_depth_table_2.depthUnits = 1000  # mm
print(current_std_depth_table_2)

advnc_mode_2.set_depth_table(current_std_depth_table_2)
config_2.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)



# Start cameras pipelines using configuration provided earlier
cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.depth)
intr_right = profile.as_video_stream_profile().get_intrinsics()
cfg_2 = pipeline_2.start(config_2)
profile_2 = cfg_2.get_stream(rs.stream.depth)
intr_left = profile_2.as_video_stream_profile().get_intrinsics()





# GUI
# Creating canvases to display in the interface
canvas_depth = tb.Canvas(master=real_time_frame,width=int(native_width/2), height=int(native_height/2))
canvas_depth.grid(row=0, column=0, padx=130, pady=5, sticky="nsew")
canvas_RGB = tb.Canvas(master=real_time_frame,width=int(native_width/2), height=int(native_height/2))
canvas_RGB.grid(row=1, column=0, padx=130, pady=5, sticky="nsew")
frame_RGB = None
frame_RGB_2 = None
frame_depth = None
frame_depth_2 = None
photo_RGB = None
photo_RGB_2 = None
photo_depth = None
photo_depth_2 = None
color_image = None
color_image_2 = None


real_time_frame.columnconfigure(0, weight=1)
real_time_frame.columnconfigure(1, weight=1)
real_time_frame.rowconfigure(0, weight=1)
real_time_frame.rowconfigure(1, weight=1)



# Create a colorizer obj
# a colorizer object assigns a color to a depth value in the depth frame in order to create a colorized depth frame
# that of course, is not equally to an RGB frame
colorizer = rs.colorizer()


# Create queues to store frames
depth_queue = create_depth_queue()
depth_queue_2 = create_depth_queue()
rgb_queue = create_rgb_queue()
rgb_queue_2 = create_rgb_queue()


#Save the frame to verify
depth_frame_to_analyze = None
rgb_frame_to_analyze = None

# Store the frame that respect the face detectors (use them in acquisition)
rgb_frame_verified = None
depth_frame_verified = None

#Store photoimage
photo_image = None


#Save photo image to show on confirmation
photo_to_show = None


#A flag that allowes to get last frames of queues 
flag = False


#Counter for face algorithms computations
counter = 0 


#Dictionary to store if threads is starting 
starts = {
    "frontal_face":False,
    "neutral_face":False,
    "far_face":False
}


#Dictionarie to store threads results
results = {
    "frontal_face":None,
    "neutral_face":None,
    "far_face":None
}


# Define threads
threads = {
    "frontal_face": None,
    "neutral_face": None,
    "far_face": None
}


#Trigger to start or less open camera
trigger = False

#Threads functions:
def is_frontal_face_thread(image):
    global results
    results["frontal_face"] = is_frontal_face(image)
        
    
def is_neutral_face_thread(image):
    global results
    results["neutral_face"] = is_neutral_face(image)
    
    
def is_far_from_camera_thread(matrix):
    global results
    results["far_face"] = is_far_from_camera(matrix)

     

#Setting threads
def set_threads(threads, depth_frame, rgb_frame):
            threads["far_face"] = tr.Thread(target=is_far_from_camera_thread, args=(depth_frame,))
            threads["frontal_face"] = tr.Thread(target = is_frontal_face_thread, args=(rgb_frame,))
            threads["neutral_face"] = tr.Thread(target=is_neutral_face_thread, args=(rgb_frame,))



#Starting threads
def start_threads(results, starts, threads):
         for value in results:
            if results[value] is None:
               if not starts[value]:
                  print(f"Main thread - starting the {value} thread...")
                  threads[value].start()
                  starts[value] = True 
                        
            else:
               print(f"Main thread - reading the result of the {value} thread...")
               starts[value] = False
               threads[value].join()    


# Wait for frames for pipeline set before and use the align obj to sync depth with color
def pipelines_config():
    global pipeline, pipeline_2, align
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    frames_2 = pipeline_2.wait_for_frames()
    frames_2 = align.process(frames_2)
    return frames, frames_2 



# Process frames and adding queues:
def process_frame(frames, frames_2):
    global photo_RGB, photo_RGB_2, photo_depth, photo_depth_2, depth_queue, depth_queue_2, rgb_queue, rgb_queue_2, canvas_depth, canvas_RGB, color_image, color_image_2, frame_depth, frame_depth_2, frame_RGB, frame_RGB_2, photo_image, native_width, native_height
 
    
    # Extract depth frames and color frames from each pipeline camera's
    depth_frame = frames.get_depth_frame()
    depth_frame_2 = frames_2.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_frame_2 = frames_2.get_color_frame()

    # Using the colorizer obj to give color to the depth image and convert it, get also the RGB frame
    depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    depth_image_2 = np.asanyarray(colorizer.colorize(depth_frame_2).get_data())
    

    # Convert color frames and colorized depth frames into RGB format
    color_image = np.asanyarray(color_frame.get_data())
    color_image_2 = np.asanyarray(color_frame_2.get_data())
    

    # Create PhotoImage to be able to display them in GUI
    frame_RGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    frame_RGB_2 = cv2.cvtColor(color_image_2, cv2.COLOR_BGR2RGB)
    frame_depth = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
    frame_depth_2 = cv2.cvtColor(depth_image_2, cv2.COLOR_BGR2RGB)
    
    
    #Save photo_RGB
    photo_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_RGB))

        # Update canvas GUI with resized frames
    photo_RGB = PIL.ImageTk.PhotoImage(
        image=PIL.Image.fromarray(frame_RGB).resize((int(native_width*70/100), int(native_height*51/100)))
    )
    photo_depth = PIL.ImageTk.PhotoImage(
        image=PIL.Image.fromarray(frame_depth).resize((int(native_width*70/100), int(native_height*51/100)))
    )
    

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    canvas_RGB.create_image(0, 0, image=photo_RGB, anchor=NW)
    canvas_depth.create_image(0, 0, image=photo_depth, anchor=NW)


   
    # Compose max 90 frames depth queue
    depth_queue.add_frame(np.asanyarray(depth_frame.get_data()))
    depth_queue_2.add_frame(np.asanyarray(depth_frame_2.get_data()))
    
    
    # Compose RGB queues (1 frame)
    rgb_queue.add_frame(np.asanyarray(color_image))
    rgb_queue_2.add_frame(np.asanyarray(color_image_2))



#Process frames and updating canvases
def update_real_time():
    global main_window
    frames, frames_2 = pipelines_config()
    process_frame(frames, frames_2)
    main_window.after(20, update_real_time)

    

#Trigger 
def go_open_camera():
    global trigger, flag, detection_button, stop_detection_button, main_window
    stop_detection_button.config(state ="enabled")
    main_window.bind("s", lambda _: stop_open_camera())
    main_window.bind("S", lambda _: stop_open_camera())
    detection_button.config(state= "disabled")
    main_window.unbind("r")
    main_window.unbind("R")
    flag = False
    trigger = True   
    open_camera() 
    
    
def stop_open_camera():
    global trigger, flag, detection_button, stop_detection_button, main_window, meter, threads, results, starts, label_distance_face, label_found_face, label_frontal_face, label_neutral_face, acquire_button, build_button
    stop_detection_button.config(state ="disabled")
    detection_button.config(state= "enabled")
    meter.configure(subtext = "Stopped")
    meter.configure(amountused = 0)
    main_window.bind("r", lambda _: go_open_camera())
    main_window.bind("R", lambda _: go_open_camera())
    main_window.unbind("s")
    main_window.unbind("S")
    set_default_labels(label_neutral_face, label_found_face, label_frontal_face, label_distance_face)
    initialize(threads, results, starts, label_distance_face, label_found_face, label_frontal_face, label_neutral_face, acquire_button, build_button)
    flag = False
    trigger = False    
    

# Function to continuous capture frames from the cameras, process them, and update the GUI elements
def open_camera():
    global trigger, counter, main_window, starts, results, threads, flag, photo_image , photo_to_show,rgb_frame_verified, depth_frame_verified, rgb_frame_to_analyze, depth_frame_to_analyze, depth_queue, depth_queue_2, rgb_queue, rgb_queue_2, results, label_neutral_face, label_found_face, label_frontal_face, label_distance_face, acquire_button, build_button, detection_button, stop_detection_button, meter
    
    if trigger == True:
          #Start computational counter
          counter += 1
        
          #Monitoring computatiions
          print(f"Computation number: {counter}\n")
        
          if flag == False:
             #Reset, take a new frame
             meter.configure(amountused = 0)
             meter.configure(subtext = "I'm analysing")
             set_default_labels(label_neutral_face, label_found_face, label_frontal_face, label_distance_face)
             initialize(threads, results, starts, label_distance_face, label_found_face, label_frontal_face, label_neutral_face, acquire_button, build_button)
             unbind_event_handlers(main_window) 
             buttons_shut(acquire_button, build_button)
        
             flag = True
            
             # Get a copy of the last frame of the queues
             depth_frame_to_analyze = depth_queue.get_last_frame()
             rgb_frame_to_analyze = rgb_queue.get_last_frame()
             
             #Apply crop and zoom for distances issues 
             crop_into_the_center(rgb_frame_to_analyze, 2)
             zoom_in(rgb_frame_to_analyze,4)
             
            
             #Threds setup
             set_threads(threads, depth_frame_to_analyze, rgb_frame_to_analyze)
   
          #Thread go 
          start_threads(results, starts, threads)
          
          """
          #Prints results   
          log_status(results)
          """
    
          # Convert dictionary values to a NumPy array
          values_array = np.array(list(results.values()))    
            
          #Avoid for cycle for complexity        
          if np.all(values_array != None):
               result = update_gui(main_window, results["neutral_face"], results["frontal_face"], results["far_face"], label_neutral_face, label_found_face, label_frontal_face, label_distance_face, meter)
               set_none(results, "far_face", "frontal_face", "neutral_face")
              
               # Store positive frames
               if result:
                    #Store rgb and depth frames passed analysis 
                    rgb_frame_verified = rgb_frame_to_analyze
                    depth_frame_verified = depth_frame_to_analyze
                    buttons_trigger(acquire_button, build_button)
                    bind_event_handlers(main_window)
                    main_window.bind("r", lambda _: go_open_camera())
                    main_window.bind("R", lambda _: go_open_camera())
                    main_window.unbind("s")
                    main_window.unbind("S")
                    stop_detection_button.config(state="disabled")
                    detection_button.config(state="enabled")
                    detection_button["text"] = "RUN AGAIN"
                    meter.configure(subtext = "Completed, \n you can acquire")
                    print(f"The final valor of the computation is {result} and the frame is stored")
                    print(f"The computation is terminated successfully in {counter} iterations\n")
                    photo_to_show = photo_image
                    rgb_frame_to_analyze = None
                    depth_frame_to_analyze = None
                    flag = False
                    counter = 0
                    return
             
               else:
                    rgb_frame_verified = None
                    depth_frame_verified = None
                    rgb_frame_to_analyze = None
                    depth_frame_to_analyze = None
                    flag = False
                    print(f"The final valor of the computation is {result} and the frame is not stored")
                    print(f"The computation is terminated not successfully in {counter} iterations\n")
                    counter = 0 
                 
                    
            
          # Schedule every 25 milliseconds
          main_window.after(25, open_camera)
    
    else:
        return   
  
  

#Function to calibrate the cameras, in order to obtain a fhiltering previous acquisition 
#we work on the depth measurements of realsense cameras
def calibrate():
    global calibrate_button, main_window

    #If calibrate() is triggered i do not want that button to be active still
    calibrate_button.config(state="disabled")
    main_window.unbind("<c>")
    main_window.unbind("<C>")
    
    # Create a new window
    calibrate_window =  tb.Toplevel(main_window)
    calibrate_window.title("Calibration")
    res = "480x600"
    calibrate_window.geometry(res)
  
    #Create a completition meter
    c_meter = tb.Meter(master= calibrate_window, metersize=350, amountused=0, bootstyle="success", textright="%",  metertype="full", subtext="Calibration...", subtextfont="Arial", amounttotal=100, stepsize=10)
    c_meter.place(anchor="center", x=240, y=300)
    

    #Crete a timestamp to store the calibrations date
    datestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.mkdir("./calibrations/" + datestring)

    
    #Calculate mean, variance from the first depth queue (of course, as a tensor)
    # in oder to do a deep filtering ro remove the noise among frames
    (
        mean_left,
        variance_left,
        inliers_left,
    ) = mean_variance_and_inliers_along_third_dimension_ignore_zeros(
        depth_queue.get_frames_as_tensor()
    )

    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(mean_left, interpolation='nearest')
    plt.show()

    plt.figure()
    plt.imshow(variance_left, interpolation='nearest')
    plt.show()


    
    #Then, convert them into numpy arrays 
    mean_left = np.asarray(mean_left.numpy())
    variance_left = np.asarray(variance_left.numpy())
    inliers_left = np.asarray(inliers_left.numpy())
    
    #Calculate mean, variance from the second depth queue (of course, as a tensor)
    (
        mean_right,
        variance_right,
        inliers_right,
    ) = mean_variance_and_inliers_along_third_dimension_ignore_zeros(
        depth_queue_2.get_frames_as_tensor()
    )
    
    meter_update(c_meter, 20)

    #Then, convert them into numpy arrays 

    mean_right = np.asarray(mean_right.numpy())
    variance_right = np.asarray(variance_right.numpy())
    inliers_right = np.asarray(inliers_right.numpy())



    #Once obtaining the variances and the means in tensor, we got the maps for variancens and means images
    #for both cameras 
    variance_image_l, zero_variance_image_l, threshold_l, filtered_means_l = get_maps(
        variance_left,
        mean_left,
        threshold=None
    )
    variance_image_r, zero_variance_image_r, threshold_r, filtered_means_r = get_maps(
        variance_right,
        mean_right,
        threshold=None
    )
    meter_update(c_meter, 10)
   
    #RealSense cameras, often provide not only depth values but also an estimate of the variance or uncertainty associated
    #with each depth measurement. This variance represents how confident the sensor is in the accuracy of a given depth value.
    #Depth measurements can sometimes be noisy or uncertain, leading
    #to errors; since a variance of about 255 often indicates that the sensosr
    #has low confidence in the depth measurements in that pixel,
    #we filter out that value to improve the point cloud reducing noise.
    #marking those pixel to zero as unreliable.
    
    
    #Once unreliable depth values set to 0, we apply filters to remove those points from the point clouds,
    #those points infact will be treaten as outliers
    
    #This way we improve accuracy of the depth data used in 3D mesh recostruction 
    
    #Get indexes as array of variance tensor image values equal 255
    indexes = np.argwhere(variance_image_l == 255)
    
    #Copy mean left tensor and replace depth values to 0 using indexed array
    selected_m_l = np.copy(mean_left)
    selected_m_l[indexes[:, 0], indexes[:, 1]] = 0
    
    #Write what we have obtained as a np array 16 bit unsigned for each pixel, (First Camera) 
    #Depth values can be quite large (up to a few thousand millimeters),
    #and using an unsigned integer type ensures that negative values are not accidentally 
    #introduced when saving or manipulating the data.
    cv2.imwrite("temp/leftDepth.png", np.uint16(selected_m_l))
    
    #Write the color image 
    cv2.imwrite("temp/leftColor.jpg", color_image)
    
    meter_update(c_meter, 20)
    indexes = np.argwhere(variance_image_r == 255)
    selected_m_r = np.copy(mean_right)
    selected_m_r[indexes[:, 0], indexes[:, 1]] = 0
    
    #Save what we have obtained as a np array 16 bit unsigned for each pixel, (Second Camera) 
    cv2.imwrite("temp/rightDepth.png", np.uint16(selected_m_r))
    
    #Save the color image 
    cv2.imwrite("temp/rightColor.jpg", color_image_2)
    
    meter_update(c_meter, 10)
    #Read the saved images for further processing 
    depth_raw_left = o3d.io.read_image("temp/leftDepth.png")
    color_raw_left = o3d.io.read_image("temp/leftColor.jpg")
    depth_raw_right = o3d.io.read_image("temp/rightDepth.png")
    color_raw_right = o3d.io.read_image("temp/rightColor.jpg")
    
    #Create RGBD images from those 
    rgbd_image_left = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw_left, depth_raw_left
    )
    rgbd_image_right = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw_right, depth_raw_right
    )

    #Set cameras instrinsic parameters (found before)
    camera_intrinsic_left = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(
            intr_left.width,
            intr_left.height,
            intr_left.fx,
            intr_left.fy,
            intr_left.ppx,
            intr_left.ppy,
        )
    )
    camera_intrinsic_right = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(
            intr_right.width,
            intr_right.height,
            intr_right.fx,
            intr_right.fy,
            intr_right.ppx,
            intr_right.ppy,
        )
    )

    #Create a point cloud using cameras instrinsic and image obtained from filtering tensor 
    pcd_left = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_left, camera_intrinsic_left
    )
    o3d.visualization.draw_geometries([pcd_left])
    pcd_right = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_right, camera_intrinsic_right
    )
    
    meter_update(c_meter, 20)
    #Set bound for cropping the point clouds rapresenting
    #the region in 3D space from which the point cloud will be retained 
    bounds = [
        [-math.inf, math.inf],   #Any coordinates for x
        [-math.inf, math.inf],   #Any coordinates for y
        [0.2, 0.7],              #Range that restrics depth values discarding the out of range #may need to change that 
    ] 
    
    #Create a bounding box
    bounding_box_points = list(itertools.product(*bounds))  
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points)
    )  

    #Crop the point cloud using the bounding box:
    pcd_left = pcd_left.crop(bounding_box)
    pcd_right = pcd_right.crop(bounding_box)

    #Rotation of 180° in X axes (because of pinhole camera)
    angolo = np.pi
    trans_x = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(angolo), -np.sin(angolo), 0.0],
            [0.0, np.sin(angolo), np.cos(angolo), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    #Apply the rotation matrix 
    pcd_left.transform(trans_x)
    pcd_right.transform(trans_x)

 

    #Rotation of 30% degrees in Y axes (because the 2 cameras are desposed in 30°)
    #to have the chance to get more points, obtaining a more detalied 3D representation
    angolo = np.pi / 6
    trans_y = np.asarray(
        [
            [np.cos(angolo), 0.0, np.sin(angolo), -0.34],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(angolo), 0.0, np.cos(angolo), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
 
   

    source = pcd_left
    target = pcd_right
    # target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=400))
    # target.orient_normals_consistent_tangent_plane(50)
    threshold = 0.01
    trans_init = np.asarray(
        [
            [np.cos(angolo), 0.0, -np.sin(angolo), -0.31],
            [0.0, 1.0, 0.0, 0.0],
            [np.sin(angolo), 0.0, np.cos(angolo), -0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    meter_update(c_meter, 10)

    #Assesses the quality of alignment between point cloud source (left) and point cloud target (right)
    #with given threshold 
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )
    print(evaluation)
    
    #Perform the iterative closest point algoritm between the 2 point clouds 
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=1000000, relative_rmse=0.001
        ),
    )
    print(reg_p2p)
    
    #Caibrated matrix obtained from ICP that rapresent the transformation needed to align the two point clouds.
    calibrated_matrix = reg_p2p.transformation

    #Saved the matrix to use in the acquire() function
    pickle.dump(calibrated_matrix, open("temp/cal_mat.mat", "wb"))


    meter_update(c_meter, 10)
    calibrate_window.update()
    c_meter.configure(subtext = "Completed")
    time.sleep(1)
    calibrate_window.destroy()

    #Re active all
    calibrate_button.config(state="enabled")
    main_window.bind("<c>", lambda _: calibrate())
    main_window.bind("<C>", lambda _: calibrate())

   




# Here is a resume of the calibrate() function:

# • Calculating mean and variance from depth queue (got as a tensor)
#  therefore we got stored depth frames in a queue and then, we have
#  the tensor who describes how the variance among depth frames change


# • Creating Variance and mean images used to identify unrealible depth values 
# exceeding 255 and set them to 0

# • Remove outliers set the exceeding pixel to zero in order to improve the accuracy of the depth data 
# we are able to get a more precise point cloud

# • Save filtered depth and color images 

# • Creating RGBD images combining saved color and depth images 

# • Setting camera instrinsic (camera left and right)
# those parameteers are: width, height, focal leght

# • Creating the point clouds from a filtered depth image,
# rotated to avoid the pinhole effects and to cope to the 30 angle degree of the cameras

# • Cropping the point clouds using a bounding box

# • Applying a transformation matrix to the point clouds

# • Performing ICP between 2 point clouds 

# • Saving calibrated trasformation matrix that represents the transformation required to align 2 point clouds 







#Function to acquire 3D mesh from point clouds 
def acquire(mesh=False):
    global main_window, acquire_button, build_button

    # Create a new window
    acquire_window =  tb.Toplevel(main_window)
    acquire_window.title("Acquiring")
    res = "480x600"
    acquire_window.geometry(res)
  
    #Create a completition meter
    a_meter = tb.Meter(master= acquire_window, metersize=350, amountused=0, bootstyle="success", textright="%",  metertype="full", subtext="Acquisition...", subtextfont="Arial", amounttotal=100, stepsize=10)
    a_meter.place(anchor="center", x=240, y=300)

    #If acquire() is triggered i do not want that button to be active still
    acquire_button.config(state="disabled")
    build_button.config(state="dibabled")
    main_window.unbind("<a>")
    main_window.unbind("<A>")
    main_window.unbind("<b>")
    main_window.unbind("<B>")
    main_window.unbind("Return")


    #Crete a timestamp to store the acquisitions
    datestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.mkdir("./acquisitions/" + datestring)
    
    #Load the transform matrix previous created 
    matrice_calibrazione = pickle.load(open("temp/cal_mat.mat", "rb"))

    #Get the last depth frame and color frame from queues 
    depth_frame_l = depth_queue_2.get_last_frame()
    color_frame_l = rgb_queue_2.get_last_frame()
    print(depth_frame_l.dtype)
    
    meter_update(a_meter, 10)

    depth_frame_r = depth_queue.get_last_frame()
    color_frame_r = rgb_queue.get_last_frame()

 
    #Save those images in a directory
    cv2.imwrite(f"./acquisitions/{datestring}/d_l.png", depth_frame_l)
    cv2.imwrite(f"./acquisitions/{datestring}/rgb_l.png", color_frame_l)
    cv2.imwrite(f"./acquisitions/{datestring}/d_r.png", depth_frame_r)
    cv2.imwrite(f"./acquisitions/{datestring}/rgb_r.png", color_frame_r)
     
    meter_update(a_meter, 10)
    #Read those images with o3d
    depth_raw_left = o3d.io.read_image(f"./acquisitions/{datestring}/d_l.png")
    color_raw_left = o3d.io.read_image(f"./acquisitions/{datestring}/rgb_l.png")
    depth_raw_right = o3d.io.read_image(f"./acquisitions/{datestring}/d_r.png")
    color_raw_right = o3d.io.read_image(f"./acquisitions/{datestring}/rgb_r.png")

    #Create RGBD images 
    rgbd_image_left = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw_left, depth_raw_left, convert_rgb_to_intensity=False
    )
    rgbd_image_right = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw_right, depth_raw_right, convert_rgb_to_intensity=False
    )
    
    meter_update(a_meter, 10)
    #Set cameras instrinsic parameters (found before)
    camera_intrinsic_left = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(
            intr_left.width,
            intr_left.height,
            intr_left.fx,
            intr_left.fy,
            intr_left.ppx,
            intr_left.ppy,
        )
    )
    camera_intrinsic_right = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(
            intr_right.width,
            intr_right.height,
            intr_right.fx,
            intr_right.fy,
            intr_right.ppx,
            intr_right.ppy,
        )
    )

    #Create a point cloud using cameras instrinsic and image obtained from filtering tensor 
    pcd_left = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_left, camera_intrinsic_left
    )
    pcd_right = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image_right, camera_intrinsic_right
    )


    #Set bound for cropping the point clouds rapresenting
    #the region in 3D space from which the point cloud will be retained 
    bounds = [
        [-math.inf, math.inf],   #Any coordinates for x
        [-math.inf, math.inf],   #Any coordinates for y
        [0.2, 0.7],              #Range that restrics depth values discarding the out of range #may need to change that 
    ]
    
    #Create a bounding box
    bounding_box_points = list(itertools.product(*bounds))  
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points)
    )  
 
    meter_update(a_meter, 10)
    # Crop the point cloud using the bounding box:
    pcd_left = pcd_left.crop(bounding_box)
    pcd_right = pcd_right.crop(bounding_box)


   #Rotation of 180° in X axes (because of pinhole camera)
    angolo = np.pi
    trans_x = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(angolo), -np.sin(angolo), 0.0],
            [0.0, np.sin(angolo), np.cos(angolo), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    #Apply the rotation
    pcd_left.transform(trans_x)
    pcd_right.transform(trans_x)

    #Apply the matrix to align the point clouds getting an unique point cloud 
    pcd_left.transform(matrice_calibrazione)
    
    #Save them
    save_pcl(pcd_left, pcd_right, datestring)
    meter_update(a_meter, 30)
    
    #Generating a 3D mesh if meth flag = true 
    if mesh:
        
        #Read gthe saved point clouds
        pcd = o3d.io.read_point_cloud(f"./acquisitions/{datestring}/pcl.pcd")
        
        #Remove outliers 
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=600, std_ratio=2.0)
        inlier_cloud = pcd.select_by_index(ind)
        inlier_cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        
        #Estimate normals of point cloud's nodes and orient them
        inlier_cloud.estimate_normals()
        inlier_cloud.orient_normals_consistent_tangent_plane(100)

        #Generating a mesh with poisson  {point clouds alignes --> mesh}
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                inlier_cloud, depth=10
            )
            
        #Filter the mesh     
        densities = np.asarray(densities)
        density_mesh = o3d.geometry.TriangleMesh()
        density_mesh.vertices = mesh.vertices
        density_mesh.triangles = mesh.triangles
        density_mesh.triangle_normals = mesh.triangle_normals
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        
        #Save the final mesh 
        o3d.io.write_triangle_mesh(f"./acquisitions/{datestring}/mesh.obj", mesh)
        meter_update(a_meter, 30)
        time.sleep(1)
        acquire_window.update()
        acquire_window.destroy()



update_real_time()



# Here's a compendium:

# calibrate() function is responsible for calibrating the 2 cameras, filtering thje depth queue to remove outliers and point unrealiable
# then, getting the transform alignemnt matrix to apply later in order to align the point clouds from camera left
# and camera right.
#The main focus of calibrate() is preprocessing, and filtering for noise reduction operating on the depth queue


#acquire() is responsible to capture synch depth and color frames and perform alignment of point clouds
# using the acquire() calibration matrix, preparing the data for 3D recostruction
# the mesh is obtained with Poisson 
#This function get the color frame and depth frame from queues and the alignemnt matrix from acquire()









#### GUI
#Call acquisition
def go_for_it(secondary_window):
    secondary_window.destroy()
    acquire(mesh=True)
    


#Dont acquire
def do_not(secondary_window):
    global rgb_frame_verified, depth_frame_verified
    
    #Since we have choosen do not go for building, we have to delete previous frame selected from facial computing
    rgb_frame_verified = None
    depth_frame_verified = None
    secondary_window.destroy()
    
    
#Second level window     
def show_confirmation_window():
    global photo_to_show, main_window, acquire_button, build_button
    
    
    #Since this functions is called, i do not want that buton to be active still
    main_window.unbind("<A>")
    main_window.unbind("<a>")
    main_window.unbind("<b>")
    main_window.unbind("<B>")
    main_window.unbind("Return")
    buttons_shut(acquire_button, build_button)

    #New assignment because photo_image is costantly updated 
    photo = photo_to_show
    
    # Create a new window
    confirm_window =  tb.Toplevel(main_window)
    confirm_window.title("Confirm")
    
    #In order to handle the frame 
    res = "1280x900"
    confirm_window.geometry(res)
  
    # Display the image on a canvas
    canvas = tb.Canvas(master=confirm_window, height=720, width=1280)
    canvas.pack(anchor="n")
    
    # Assuming photo is loaded correctly and compatible with Tkinter's PhotoImage
    # Get the dimensions of the photo
    image_width = photo.width()
    image_height = photo.height()

    # Calculate the coordinates to center the image
    x = (1280 - image_width) / 2
    y = (720 - image_height) / 2

    # Anchor image to the calculated coordinates to center it
    canvas.create_image(x, y, anchor='nw', image=photo)

    
    # Display the message
    message_frame = tb.Labelframe(master= confirm_window, text="Confirm")
    message_frame.pack(pady=10, anchor="s")

    message = tb.Label(text="Produce face mesh based on this picture?", master=message_frame, font=("Helvetica", 25), width = 35, bootstyle = "info")
    message.place(anchor="nw", x= 120)
    
    # New buttons
    confirm_button = tb.Button(message_frame, text="YES", state="enabled", width=20, style="success - outline",bootstyle = "success-outline" ,command=lambda: go_for_it(confirm_window))
    delete_button = tb.Button(message_frame, text="CANCEL", state="enabled",  width=20, style="danger - outline",bootstyle = "danger-outline" ,command=lambda: do_not(confirm_window))
    
    confirm_button.pack(side="left", padx=100, pady=60,anchor="sw")  # Positioned to the left with padding
    delete_button.pack(side="left", padx=100, pady=60, anchor="se")  # Positioned to the left of the OK button with padding
    
    #New binds
    confirm_window.bind("<y>", lambda _: go_for_it(confirm_window))
    confirm_window.bind("<Y>", lambda _: go_for_it(confirm_window))
    confirm_window.bind("<Return>", lambda _: go_for_it(confirm_window))
    confirm_window.bind("<Escape>", lambda _: do_not(confirm_window))
    
    confirm_window.mainloop()     


#Set menues 
menu_configuration(main_window) 

#Fullscreen
fullscreen(main_window, True)

# Command binds
main_window.bind("<q>", lambda _: main_window.quit())
main_window.bind("<Q>", lambda _: main_window.quit())
main_window.bind("<F11>", lambda _: fullscreen(main_window, True))
main_window.bind("<Escape>", lambda _: fullscreen(main_window, False))
main_window.bind("<c>", lambda _: calibrate())
main_window.bind("<C>", lambda _: calibrate())
main_window.bind("r", lambda _: go_open_camera())
main_window.bind("R", lambda _: go_open_camera())

main_window.mainloop()
####