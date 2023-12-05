from tkinter import *
from tkinter import messagebox
import ttkbootstrap as tb


#Fullscreen setup
def fullscreen(main_window, bool):
    #True or False 
    main_window.attributes("-fullscreen", bool)


#Help
def display_help():
    help_message = """
    Welcome to the D435 Depth Stream GUI:

    - Press "Q" to exit the application.
    - Press "F11" to go into fullscreen and "Escape" to exit from fullscreen
    - Use "Calibrate" button to start the filtering and calibrate the cameras or press "C".
    - Use "Acquire" button to acquire the face point cloud or press "A/Enter".
    - Use "Build" button to acquire the 3D face mesh or press "B/Enter".
    - Click on "Run Detections" to enable face algorithms or press "R"
    - Click on "Stop Detections to stop face algorithms or press "S"

    • Note that Acquire and Build buttons will be enabled under some requirements as shown:
      ~ Correct distance by the subject from the camera
      ~ Correct facial expression (neutral)
      ~ Face deteced properly with correct visage position and mouth closed
      
    • We will notify you if a week or more has passed since the last calibration.

    Code logic and implementation by Giuseppe Maurizio Facchi and Nicola Montagnese
    """
    
    messagebox.showinfo("Help", help_message)
    
# Menu configuration and calibration
def menu_configuration(window):
    # Create a menubar
    menu_bar = tb.Menu(window)
    window.config(menu=menu_bar)

    # Adding menus
    help_menu = tb.Menu(menu_bar, tearoff=0)
    

    # Add commands
    menu_bar.add_cascade(label="Realsense", menu=help_menu)
  
    help_menu.add_command(label="Help Contents", command=display_help)

    help_menu.add_command(label="About Realsense", 
                          command=lambda: messagebox.showinfo("About", 
                                                              "D435 Depth Stream GUI\n by Giuseppe Maurizio Facchi and Nicola Montagnese\n\nUniversità degli studi di Milano"))

    help_menu.add_separator()
    help_menu.add_command(label="Quit Realsense", 
                          command=lambda: window.quit())
    
    
# Widgets functions
def buttons_trigger(acquire_button, build_button):
    acquire_button.config(state="enabled")
    build_button.config(state="enabled")
    
    
def buttons_shut(acquire_button, build_button):
    acquire_button.config(state="disabled")
    build_button.config(state="disabled")
    
    
#Updating light    
def meter_update(meter, valor):
    meter.step(valor)  
     
     
#Set the labels to the default text
def set_default_labels(l1,l2,l3,l4):
    l1.config(text = "Checking...")
    l2.config(text = "Checking...")
    l3.config(text = "Checking...")
    l4.config(text = "Checking...")
    l1.config(bootstyle = "secondary") 
    l2.config(bootstyle = "secondary") 
    l3.config(bootstyle = "secondary") 
    l4.config(bootstyle = "secondary")   