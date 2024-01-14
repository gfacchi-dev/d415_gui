from tkinter import *
from tkinter import ttk
from utilsjacopo import *
import cv2
from PIL import Image, ImageTk
from camera import Camera
import pyrealsense2 as rs

def acquire_window():
    global window, live_feed, canvas_left, canvas_center, canvas_right
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
    live_feed.pack(expand=True, fill=BOTH)
    # Run the loop
    update_video()
    acquire_window.mainloop()

def acquire():
    global window, live_feed, canvas_left, canvas_center, canvas_right, msg_lbl, acquire_btn, schedule_id
    live_feed.after_cancel(schedule_id)
    # TODO: Richiamare la funzione get data da ogni singola istanza delle camere, ricavare pointcloud, applicare matrice trasformazione e salvare i dati

    print("Acquiring")
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
    btn_calibrate = ttk.Button(live_feed, text="Calibrate", width=40, command=calibrate_procedure)
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
    update_video()
    cal_window.mainloop()


def update_video():
    global schedule_id
    ret, frame = cap.read()
    # TODO: Richiamare la funzione get data da ogni singola istanza delle camere e inserire i dati in uno stack (forse no)
    # Quando l'utente clicca su calibrate, la funzione calibrate_procedure() viene richiamata e lo stream viene fermato
    if ret:
        # Convert the OpenCV BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
        # Convert the RGB image to a Tkinter PhotoImage
        photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        
        # Update the Tkinter canvas with the new PhotoImage
        canvas_left.create_image(0, 0, anchor=NW, image=photo)
        canvas_left.photo = photo
        canvas_center.create_image(0, 0, anchor=NW, image=photo)
        canvas_center.photo = photo
        canvas_right.create_image(0, 0, anchor=NW, image=photo)
        canvas_right.photo = photo
    # Schedule the update function to be called after a delay
    schedule_id = live_feed.after(10, update_video)

def calibrate_procedure():
    global cal_window, schedule_id
    print("Calibrating")
    live_feed.after_cancel(schedule_id)
    # TODO: usare i valori nello stack per calibrare il sistema
    acquire_btn.config(state="enabled")
    msg_lbl.config(text="Calibration completed, you can now acquire the data.")
    cal_window.destroy()

global center, left, right

SCALA = 3
colorizer = rs.colorizer()
center = Camera("Center", "210622061176", SCALA, 100, 800)
left = Camera("Left", "211222063114", SCALA, 100, 800)
right = Camera("Right", "211122060792", SCALA, 100, 800)

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
acquire_btn = ttk.Button(frm, text="Acquire", command=acquire_window)
acquire_btn.pack()
acquire_btn.config(state="disabled")
quit_btn = ttk.Button(frm, text="Quit", command=window.destroy)
quit_btn.pack()
msg_lbl = ttk.Label(frm, text="")
msg_lbl.pack()
frm.pack(expand=True, fill=BOTH)
if check_calibration():
    acquire_btn.config(state="enabled")
    M_L2C, M_R2C = get_calibration()
else:
    print("No calibration found")
    msg_lbl.config(text="No calibration found, please calibrate the system first.")
    acquire_btn.config(state="disabled")
window.mainloop()