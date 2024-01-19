from tkinter import *
from tkinter import ttk
from utilsjacopo import *
import cv2
from geometry import *
from PIL import Image, ImageTk
from camera import Camera
import pyrealsense2 as rs
import numpy as np
import copy
import open3d as o3d
from qreader import QReader


def show_acquire_window():
    global window, live_feed, canvas_left, canvas_center, canvas_right, acquire_window
    print("Open Camera")
    acquire_window = Toplevel(window)
    acquire_window.title("Separate Window")
    # Create a frame to show the live camera feed
    live_feed = ttk.Frame(acquire_window, padding=10)
    title = ttk.Label(live_feed, text="Camera", font=("Arial", 10), foreground="black")
    title.pack()
    description_label = ttk.Label(live_feed, text=acquire_text, font=("Arial", 10), foreground="black")
    description_label.pack()
    cameras = ttk.Frame(live_feed)
    canvas_left = Canvas(cameras, width=480, height=640, bg='black')
    canvas_left.pack(side="left", padx=10)
    canvas_center = Canvas(cameras, width=480, height=640, bg='black')
    canvas_center.pack(side="left", padx=10)
    canvas_right = Canvas(cameras, width=480, height=640, bg='black')
    canvas_right.pack(side="left", padx=10)
    cameras.pack(expand=True, fill=BOTH)
    execute_acquire_btn = ttk.Button(live_feed, text="Acquire", width=40, command=acquire)
    execute_acquire_btn.pack()
    live_feed.pack(expand=True, fill=BOTH)
    # Run the loop
    update_video(quadrilaterals=False)
    acquire_window.mainloop()

def acquire():
    global window, live_feed, canvas_left, canvas_center, canvas_right, msg_lbl, acquire_btn, schedule_id, center, left, right, calibration_dir
    global acquire_window
    live_feed.after_cancel(schedule_id)
    print("Acquiring")
    acquire_shot(center, left, right, calibration_dir)
    msg_lbl.config(text="Acquisition completed.")
    acquire_window.destroy()
    #acquire_window()
    
def open_calibration_window():
    global window, live_feed, canvas_left, canvas_center, canvas_right, msg_lbl, acquire_btn, schedule_id
    global cal_window

    print("Opening Calibration Window")
    cal_window = Toplevel(window)
    cal_window.title("Calibration")
    # Create a frame to show the live camera feed
    live_feed = ttk.Frame(cal_window, padding=10)
    title = ttk.Label(live_feed, text="Camera", font=("Arial", 10), foreground="black")
    title.pack()
    description_label = ttk.Label(live_feed, text=calibrate_text, font=("Arial", 10), foreground="black")
    description_label.pack()
    btn_calibrate = ttk.Button(live_feed, text="Calibrate", width=40, command=confirmation)
    btn_calibrate.pack()
    cameras = ttk.Frame(live_feed)
    canvas_left = Canvas(cameras, width=480, height=640, bg='black')
    canvas_left.pack(side="left", padx=10)
    canvas_center = Canvas(cameras, width=480, height=640, bg='black')
    canvas_center.pack(side="left", padx=10)
    canvas_right = Canvas(cameras, width=480, height=640, bg='black')
    canvas_right.pack(side="left", padx=10)
    cameras.pack(expand=True, fill=BOTH)
    live_feed.pack(expand=True, fill=BOTH)
    # Run the loop
    update_video(quadrilaterals=True)
    cal_window.mainloop()


def update_video(quadrilaterals: bool = False):
    global schedule_id, center, left, right
    if quadrilaterals:
        center_frame, _ = center.detect_quadrilaterals()
        left_frame, _ = left.detect_quadrilaterals()
        right_frame, _ = right.detect_quadrilaterals()
    else:
        center_frame = center.get_rgb_frame()
        left_frame = left.get_rgb_frame()
        right_frame = right.get_rgb_frame()
    
    if center_frame is not None and left_frame is not None and right_frame is not None:
        # Convert the OpenCV BGR image to RGB
        rgb_frame_c = cv2.cvtColor(center_frame, cv2.COLOR_BGR2RGB)
        rgb_frame_c = cv2.rotate(rgb_frame_c, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb_frame_c = cv2.resize(rgb_frame_c, (480, 640))
        photo_c = ImageTk.PhotoImage(Image.fromarray(rgb_frame_c))

        rgb_frame_l = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        rgb_frame_l = cv2.rotate(rgb_frame_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb_frame_l = cv2.resize(rgb_frame_l, (480, 640))
        photo_l = ImageTk.PhotoImage(Image.fromarray(rgb_frame_l))

        rgb_frame_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        rgb_frame_r = cv2.rotate(rgb_frame_r, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb_frame_r = cv2.resize(rgb_frame_r, (480, 640))
        photo_r = ImageTk.PhotoImage(Image.fromarray(rgb_frame_r))

        # Update the Tkinter canvas with the new PhotoImage
        canvas_left.create_image(0, 0, anchor=NW, image=photo_l)
        canvas_left.photo = photo_l
        canvas_center.create_image(0, 0, anchor=NW, image=photo_c)
        canvas_center.photo = photo_c
        canvas_right.create_image(0, 0, anchor=NW, image=photo_r)
        canvas_right.photo = photo_r
    # Schedule the update function to be called after a delay
    schedule_id = live_feed.after(10, update_video, quadrilaterals)
    

def calibrate_procedure():
    global cal_window, center, left, right, calibration_dir
    cal_window.destroy()
    compute_calibration(center=center, left=left, right=right)  
    calibration_dir = get_calibration()
    msg_lbl.config(text="Calibration completed, you can now acquire the data.")

def confirmation():
    global cal_window, schedule_id, cal_window
    print("Calibrating")
    live_feed.after_cancel(schedule_id)
    # TODO: usare i valori nello stack per calibrare il sistema
    acquire_btn.config(state="enabled")
    confirmation_button = ttk.Button(live_feed, text="CONFERMA", width=40, command=calibrate_procedure)
    confirmation_button.pack()
    

global center, left, right, calibration_dir

SCALA = 3
CLAMP_MAX = 1000
colorizer = rs.colorizer()
center = Camera("Center", "210622061176", SCALA, 100, CLAMP_MAX, True)
left = Camera("Left", "211222063114", SCALA, 100, CLAMP_MAX, False)
right = Camera("Right", "211122060792", SCALA, 100, CLAMP_MAX, False)

cap = cv2.VideoCapture(0)
window = Tk()
window.geometry("640x360")
window.iconphoto(False, PhotoImage(file='images/logo.png'))
window.title("Hello")
frm = ttk.Frame(window, padding=10)
title = ttk.Label(frm, text="Realsense", font=("Arial", 10), foreground="black")
title.pack()
acquire_btn = ttk.Button(frm, text="Calibrate", command=open_calibration_window)
acquire_btn.pack()
acquire_btn = ttk.Button(frm, text="Acquire", command=show_acquire_window)
acquire_btn.pack()
acquire_btn.config(state="disabled")
quit_btn = ttk.Button(frm, text="Quit", command=window.destroy)
quit_btn.pack()
last_cal = ttk.Label(frm, text="Last calibration: NOT FOUND")
last_cal.pack()
msg_lbl = ttk.Label(frm, text="")
msg_lbl.pack()
frm.pack(expand=True, fill=BOTH)
if check_calibration():
    acquire_btn.config(state="enabled")
    calibration_dir = get_calibration()
    last_cal.config(text=f"Last calibration: {calibration_dir}")
else:
    print("No calibration found")
    msg_lbl.config(text="No calibration found, please calibrate the system first.")
    acquire_btn.config(state="disabled")
window.mainloop()