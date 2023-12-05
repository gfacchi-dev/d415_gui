import cv2
import numpy as np
from deepface import DeepFace
import os
import datetime
import re
import mediapipe as mp
from tkinter import *
from tkinter import messagebox
from deepface import DeepFace
from utils import FrameQueue

from gui import (
    meter_update,
    set_default_labels,
    buttons_shut
)



# Check if a face is neutral
def is_neutral_face(image):    
    try:    
        # Instantiate DeepFace analyzing object with specific emotion parameter
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False, silent= True)
        
        if result:
         # Check if the result is an empty list, indicating no face detected
         if not result[0].get("dominant_emotion"):
            print("Neutral face thread says no emotion detected\n")
            return False
         
         # Get the dominant emotion in the result [list]
         dominant_emotion = result[0]["dominant_emotion"]

         # Print emotion
         print(f"DOMINANT EMOTION: {dominant_emotion}")
        
       
         # Check if the dominant emotion is neutral
         if dominant_emotion == "neutral" or dominant_emotion == "sad":
            print("Neutral face thread has finished computing and the correct emotion is found\n")
            return True
            
         else:
            print("Neutral face thread has finished computing and the incorrect emotion is found\n")
            return False
        
        else:
            print("Neutral face thread says no emotion detected\n")
            return False
        
    except ValueError as v:
        return False




#Function to calculate the euclidean distance 
def calculate_euclidean_distance(x1, x2, y1, y2):
    point1 = (x1,y1)
    point2 = (x2,y2)
    #Calculate normal of the vector that connets the points 
    return np.linalg.norm(np.array(point1) - np.array(point2))


#T face ratio
def calculate_T_ratio(horizontal_vector, vertical_vector):
    if vertical_vector == 0:
        raise ValueError("Division by zero not allowed\n")
    else:
        return horizontal_vector/vertical_vector
        

#Check T_ratio
def verify_T_ratio(treshold, ratio):
    result = treshold[0] <= ratio <= treshold[1]
    
    if result:
        return True
    
    return False


#Compute dot product, and thus the angle formed by 2 vectors perpendicolar
def get_angles_between_perpendicular_vectors(p1,p2,p3,p4,distance_A, distance_B):
    # Calculate vectors P1P3 and P2P4 using the coordinates of the points
    P1P2 = np.array(p1) - np.array(p2)
    P4P3 = np.array(p4) - np.array(p3)
    
    # Calculate dot product of P1P3 and P2P4
    dot_product = np.dot(P1P2, P4P3)

    # Calculate cosine of the angle between P1P3 and P2P4 using dot product
    cos_angle = dot_product / (distance_A * distance_B)
    
    # Calculate angle in radians
    angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    # Calculate the second angle by subtracting from 180 degrees
    second_angle_degrees = 180 - angle_degrees
    
    return angle_degrees, second_angle_degrees



#Facial landmarks with Mediapipe
main_faceLandmarks_coordinates = {
    
        "Nose_Tip": 4,
        "Upper_Face": 10,
        
        "Left_Face_Side": 35,
        "Right_Face_Side": 263,
        
        "Left_Lips":61,
        "Right_Lips":308,
        "Upper_Lips":0,
        "Bottom_Lips": 17
}



# Detect a frontal face and mouth closed
def is_frontal_face(image):
    # Instantiate the FaceMesh object with the confidence between 0 and 1,
    # 0.30, would require higher confidence for face detection and might produce fewer detections but with higher confidence, so I chose 0.30
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.30, min_tracking_confidence=0.30)

    # Convert the image to RGB (MediaPipe uses RGB format) - This process can be done on a video frame, taking one frame at a time
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    result = face_mesh.process(image_rgb)

    # Image resolution to normalize the landmark coordinates
    height, width, _ = image.shape
    
    #Store points for dot product later
    p1 = 0
    p2 = 0
    p3 = 0
    p4 = 0

    #Inizializate variables
    vertical_face_vector = 0
    horizontal_face_vector = 0
        
    # Check if at least one face is detected
    if not result or not result.multi_face_landmarks:
        print("Frontal face thread has finished computing and no face is detected\n")
        return "Cannot"
    
    else:
        # Get the landmarks for the first detected face
        landmarks = result.multi_face_landmarks[0]

        # Initialize landmarks
        nose_tip = landmarks.landmark[4]
        upper_face = landmarks.landmark[10]
        left_face = landmarks.landmark[35]
        right_face = landmarks.landmark[263]
        
        
        # Calculate Euclidean distances between points
        #Vertical face line
        vertical_face_vector = calculate_euclidean_distance(
            nose_tip.x * width, upper_face.x * width,
            nose_tip.y * height, upper_face.y * height
        )
        
        #Horizontal face line
        horizontal_face_vector = calculate_euclidean_distance(
            left_face.x * width, right_face.x * width,
            left_face.y * height, right_face.y * height
        )
        
        #I need to store the 4 points in order to evaluate the 2 perpendicolar vectors
        p1 = (nose_tip.x * width, nose_tip.y * height)
        p2 = (upper_face.x * width, upper_face.y * height)
        p3 = (left_face.x * width, left_face.y * height)
        p4 = (right_face.x * width,right_face.y * height)
        
        
         
    print(f"VERTICAL FACE LINE: {vertical_face_vector} ")
    print(f"HORIZONTAL FACE LINE: {horizontal_face_vector} ")    
    print("\n")    
       
    #Define ideal T face treshold: (Ideally, for a frontal face, this ratio should be around 1, angles comprehended)
    T_THRESHOLD = (0.98, 1.14)
   

    try:
        #Evaluate angles of T shape 
        left_angle, right_angle = get_angles_between_perpendicular_vectors(p1,p2,p3,p4,horizontal_face_vector, vertical_face_vector)

        #Evaluate ratio
        face_result = calculate_T_ratio(horizontal_face_vector, vertical_face_vector)
        angle_result = calculate_T_ratio(left_angle, right_angle)

        print(f"ANGLES OF T FACE: {left_angle, right_angle}\n")
        print(f"FACE RAIO RESULT: {face_result}")
        print(f"ANGLE RAIO RESULT: {angle_result}\n")

        #Boolean definitions
        face_checked = verify_T_ratio(T_THRESHOLD, face_result)
        angles_checked = verify_T_ratio(T_THRESHOLD, angle_result)
       
        #Conditions
        if face_checked and angles_checked:
            print("Frontal face thread says face is frontal\n")
            return True
          
        elif face_checked and not angles_checked:
            print("Frontal face thread says face is not frontal but rotated vertically\n")
            return 1
             
        elif not face_checked and angles_checked:
            print("Frontal face thread says face is not frontal but rotated horizontally\n")
            return 2
            
        else:
            print("Frontal face thread says face is not frontal but rotated horizontally")
            print("Frontal face thread says face is not frontal but rotated vertically\n")
            return False        
                        
    except ValueError as v:
        print("Frontal face thread says exception risen\n")
        print(v)
        return "Cannot"


#Verify if there is a minimun value in a matrix
def is_far_from_camera(depth_matrix):
    #Take to the array form 
    array = np.array(depth_matrix)
    
    #Check if there are positive elements in the array
    result = np.any(array > 0)

    if result:
       #Clean up 
       depth_min = np.min(array[array != 0])
      
       # Set the distance range criteria (mm)
       DISTANCE_CRITERIA_RANGE = (250, 1500)  # may need to change that

       print(f"DEPTH MIN DISTANCE: {depth_min}")

       # Compare the minimum depth value with the threshold
       result = DISTANCE_CRITERIA_RANGE[0] <= depth_min <= DISTANCE_CRITERIA_RANGE[1]

       if result:
         print("Far face thread has finished computing, and the face is at the right distance.")
         return True
     
       else:
         print("Far face thread has finished computing, and the face is not at the right distance.")
         return False
      
    else:
       print("Cannot find a valid depth matrix")
       return False



#Create queues
def create_depth_queue():
   return FrameQueue(max_frames=90, frame_shape=(720, 1280))


def create_rgb_queue():
   return FrameQueue(max_frames=1, frame_shape=(720, 1280, 3))



#Function to detect if there is a directory in a relative path 
def is_there_a_directory(path):
    return any(entry.is_dir() for entry in os.scandir(path))
  
  


#Function to check if more then a week is passed since the last calibration
def a_week_is_passed(directory_path):
    # Define a regular expression pattern to match the timestamp format
    timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}'
    
    # Create a compiled regex object
    regex = re.compile(timestamp_pattern)
    
    # Get a list of directories that match the timestamp pattern
    matching_directories = [d for d in os.listdir(directory_path) if regex.match(d) and os.path.isdir(os.path.join(directory_path, d))]
    
    # Get a list of directories that don't match the timestamp pattern
    not_matching_directories = [x for x in os.listdir(directory_path) if not regex.match(x) and os.path.isdir(os.path.join(directory_path, x))]
    
    if not matching_directories:
        return False
    
    if  not_matching_directories:
        messagebox.showwarning("Warning","Wrong directory detected in the directory 'Acquisizioni', check it out")
        return False

    # Sort the matching directories by timestamp (assuming the directory names can be compared as timestamps)
    matching_directories.sort()
    
    # Take the most recent directory, which should be the last one after sorting
    most_recent_directory_name = matching_directories[-1]

    # Parse the timestamp from the directory name
    saved_timestamp = datetime.datetime.strptime(most_recent_directory_name, "%Y-%m-%d %H-%M-%S")

    # Calculate the difference between the current date and the saved timestamp
    current_date = datetime.datetime.now()
    time_difference = current_date - saved_timestamp

    # Check if a week (7 days) or more has passed
    return time_difference.days >= 7
    



# Display information
def verify_calibration(directory_path):
    if is_there_a_directory(directory_path) == False:
        messagebox.showinfo("Information", "You have not done a calibration yet.\nPlease do a calibration before acquiring")
        return

    if is_there_a_directory(directory_path) == True and a_week_is_passed(directory_path) == True:
        messagebox.showinfo("Information", "A week or more has passed since the last calibration, we suggest you do a calibration")
        return
        
        
#Log print out
def print_out(frontal, far, neutral):
            print("\n")
            print(f"Fontal face thread returned:{frontal}")
            print(f"Far face thread returned:{far}")
            print(f"Neutral face thread returned:{neutral}\n")
          

   
# Process GUI widgets 
def update_gui(window, neutral, frontal_face, far_face, label_neutral_face, label_found_face, label_frontal_face, label_distance_face, meter):  
        print_out(frontal_face, far_face, neutral)

        if neutral:
            label_neutral_face.config(text="Face neutral")
            label_neutral_face.config(bootstyle = "success")
            meter_update(meter, 25)
            
        else:
            label_neutral_face.config(text="Face not neutral")
            label_neutral_face.config(bootstyle = "danger")
        
        
        if frontal_face == True:
            label_frontal_face.config(text="Face frontal")
            label_frontal_face.config(bootstyle = "success")
            label_found_face.config(text="Face found")
            label_found_face.config(bootstyle = "success")
            meter_update(meter, 50)
    
            
        elif frontal_face == False :
            label_frontal_face.config(text="Face is rotated vertically \nand horizontally")
            label_frontal_face.config(bootstyle = "danger")
            label_found_face.config(text="Face found")
            label_found_face.config(bootstyle = "success")
            meter_update(meter,25)
            
           
        elif frontal_face == 1:
            label_frontal_face.config(text="Face rotated vertically")
            label_frontal_face.config(bootstyle = "danger")
            label_found_face.config(text="Face found")
            label_found_face.config(bootstyle = "success")
            meter_update(meter,25)

        elif frontal_face == 2:
            label_frontal_face.config(text="Face rotated horizontally")
            label_frontal_face.config(bootstyle = "danger")
            label_found_face.config(text="Face found")
            label_found_face.config(bootstyle = "success")
            meter_update(meter,25)

        elif frontal_face == "Cannot":
            label_frontal_face.config(text="Cannot verify")
            label_frontal_face.config(bootstyle = "danger")
            label_found_face.config(text="Face not found")
            label_found_face.config(bootstyle = "danger")    
             
        
        if far_face:
            label_distance_face.config(text="Correct distance")
            label_distance_face.config(bootstyle = "success")
            meter_update(meter, 25)
            
            
        else:
            label_distance_face.config(text="Incorrect distance")
            label_distance_face.config(bootstyle = "danger")
      
        

        # Check if all conditions are met
        if neutral and frontal_face == True and far_face:
            window.update()
            return True
        
        else:
            window.update()
            return False
        


def all_elements_not_none(dictionary):
    # Check if the dictionary has exactly 5 keys
    if len(dictionary) != 4:
        return False

    # Iterate through the values in the dictionary
    for value in dictionary.values():
        if value is None:
            return False

    # If all values are not None, return True
    return True  



def all_elements_false(dictionary):
    # Check if the dictionary has exactly 5 keys
    if len(dictionary) != 4:
        return False

    # Iterate through the values in the dictionary
    for value in dictionary.values():
        if value is True:
            return False

    # If all values are not None, return True
    return True  



def dictionary_false(results):
    for key in results:
        results[key] = None


#Restore frame
def image_restoration(rgb_frame):

 # Apply histogram equalization for enhancing contrast
 equalized_image = cv2.equalizeHist(rgb_frame)

  # Apply Gaussian blur for noise reduction
 blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
 
 # Convert the image to grayscale
 gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

 # Apply edge enhancement
 edges = cv2.Canny(gray_image, 30, 70)
 edge_enhanced_image = cv2.bitwise_and(blurred_image, blurred_image, mask=edges)
 
 # Apply median filter for noise reduction
 filtered_image = cv2.medianBlur(edge_enhanced_image, 5)  

 # Convert the grayscale image back to RGB format
 return cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)


#Set none without cycling
def set_none(dictionary, string1, string2, string3):
        dictionary[string1] = None
        dictionary[string2] = None
        dictionary[string3] = None

        
#Set false without cycling
def set_false(dictionary, string1, string2, string3):
        dictionary[string1] = False
        dictionary[string2] = False
        dictionary[string3] = False



#Print all values 
def log_status(dic):
    print(f"\nSo far i have gathered this results:")
    for keys, values in dic.items():
        print(keys + ":" + str(values))
    print("\n")
    

#Crop 
def crop_into_the_center(image, crop_factor):
    # Get the height and width of the original image
    height, width = image.shape[:2]

    # Calculate the cropping dimensions
    crop_height = int(height * crop_factor)
    crop_width = int(width * crop_factor)

    # Calculate the cropping coordinates to move towards the center
    crop_x = max(0, (width - crop_width) // 2)
    crop_y = max(0, (height - crop_height) // 2)

    # Crop the image
    image = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

    

#Zoom
def zoom_in(image, zoom_factor):
    # Get the height and width of the original image
    height, width = image.shape[:2]

    # Calculate the new height and width after zooming
    zoomed_height = int(height * zoom_factor)
    zoomed_width = int(width * zoom_factor)

    # Resize the image using bicubic interpolation
    image = cv2.resize(image, (zoomed_width, zoomed_height), interpolation=cv2.INTER_CUBIC)
    
    
#Take back the system to state 0    
def initialize(threads, results, starts, l1,l2,l3,l4, acquire_button, build_button):
        set_none(threads, "frontal_face", "neutral_face","far_face")
        set_none(results,  "frontal_face", "neutral_face","far_face")
        set_false(starts, "frontal_face", "neutral_face","far_face")
        set_default_labels(l1,l2,l3,l4)
        buttons_shut(acquire_button, build_button)