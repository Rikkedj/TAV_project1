#**************************************************************************************

#

#   Driver Monitoring Systems using AI (code sample)

#

#   File: eyes_position.m

#   Author: Jacopo Sini

#   Company: Politecnico di Torino

#   Date: 19 Mar 2024

#

#**************************************************************************************



# 1 - Import the needed libraries

import cv2

import mediapipe as mp

import numpy as np 

import time

import statistics as st

import os

# 1.5 - Define functions

def drowsiness_detection(EAR_right, EAR_left, driver_asleep, time_start_drowsy, time_asleep):
    # Drowsiness detection

    EAR_treshold = 0.2
    left_eye_closed = False
    right_eye_closed = False


    if EAR_right < EAR_treshold:
        right_eye_closed = True
  
    if EAR_left < EAR_treshold:
        left_eye_closed = True

    if right_eye_closed and left_eye_closed:
        if driver_asleep == False:
            print("Drowsiness detected")
            cv2.putText(image, "Drowsiness detected", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            time_start_drowsy = time.time()
            driver_asleep = True
        elif driver_asleep == True:
            time_asleep = time.time() - time_start_drowsy

    if not right_eye_closed and not left_eye_closed:
        driver_asleep = False
        time_asleep = 0

    if time_asleep > 10:
        print("Driver asleep")      # !!! ALARM !!!
        cv2.putText(image, "Driver asleep", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return driver_asleep, time_start_drowsy, time_asleep




# 2 - Set the desired setting

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(

    max_num_faces=1,

    refine_landmarks=True, # Enables  detailed eyes points

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

)

mp_drawing_styles = mp.solutions.drawing_styles

mp_drawing = mp.solutions.drawing_utils



drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# Get the list of available capture devices (comment out)

#index = 0

#arr = []

#while True:

#    dev = cv2.VideoCapture(index)

#    try:

#        arr.append(dev.getBackendName)

#    except:

#        break

#    dev.release()

#    index += 1

#print(arr)



# 3 - Open the video source

cap = cv2.VideoCapture(0) # Local webcam (index start from 0)
max_EAR_right = 0.01
max_EAR_left = 0.01
driver_asleep = False
time_start_drowsy = 0
time_asleep = 0
all_points_left_eye = False
all_points_right_eye = False

# 4 - Iterate (within an infinite loop)

while cap.isOpened(): 

    

    # 4.1 - Get the new frame

    success, image = cap.read() 

    

    start = time.time()



    # Also convert the color space from BGR to RGB

    if image is None:

        break

        #continue

    #else: #needed with some cameras/video input format

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    # To improve performace

    image.flags.writeable = False

    

    # 4.2 - Run MediaPipe on the frame

    results = face_mesh.process(image)



    # To improve performance

    image.flags.writeable = True



    # Convert the color space from RGB to BGR

    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



    img_h, img_w, img_c = image.shape

    # making new arrays for each frame in the "video", calcluating the points for each frame, and then calculating the gaze for each frame. If we want to calculate the gaze more often, we have to watch when we convert the arrays to numpy arrays because numpy arrays do not have the "append" function.
    face_2d = []
    face_3d = []
    right_eye_2d = []
    right_eye_3d = []
    left_eye_2d = []
    left_eye_3d = []


    point_RER = [] # Right Eye Right

    point_REB = [] # Right Eye Bottom

    point_REL = [] # Right Eye Left

    point_RET = [] # Right Eye Top



    point_LER = [] # Left Eye Right

    point_LEB = [] # Left Eye Bottom

    point_LEL = [] # Left Eye Left

    point_LET = [] # Left Eye Top



    point_REIC = [] # Right Eye Iris Center

    point_LEIC = [] # Left Eye Iris Center

    p1_right = []
    p2_right = []
    p3_right = []
    p4_right = []
    p5_right = []
    p6_right = []
    p1_left = []
    p2_left = []
    p3_left = []
    p4_left = []
    p5_left = []
    p6_left = []
    
    # 4.3 - Get the landmark coordinates

    if results.multi_face_landmarks:        # multi_face_landmarks is a list of potentially multiple faces that is detected. if resukts.multi..., if this if statement is true, at least one face is detected. (The maximum from the settings is now 1, so....)

        for face_landmarks in results.multi_face_landmarks:     # face_landmarks is all landmarks (points) from one of the detected faces. I.e. here we are iterating through the faces, most of the times, we will only have one face. This line is kind of useless now that the maximum number of allowed faces is 1, but anyways

            for idx, lm in enumerate(face_landmarks.landmark):  # "idx" is the id (number) of the landmark, the one in the drawings, and "lm" is coordinate points. Here we are iterating through all landmarks, i.e. 477 points, one at a time 


                # ----------------- POINTS -----------------

                #LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
                #LEFT_EYE_BORDER=[ 473, 362, 374, 263, 386 ]

                #RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
                #RIGHT_EYE_BORDER=[ 468, 33, 145, 133, 159 ]

                #LEFT_IRIS  = [ 473, 474, 475, 476, 477 ] 
                #RIGHT_IRIS = [ 468, 469, 470, 471, 472 ]

                # ----------------- GET POINTS FOR RIGHT EYE DRAWING -----------------

                if idx == 33:
                    point_RER = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 145:
                    point_REB = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                
                if idx == 144:
                    point_REBL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                
                if idx == 153:
                    point_REBR = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
        
                if idx == 133:
                    point_REL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 159:
                    point_RET = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 158:
                    point_RETR = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 160:
                    point_RETL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)


                # ----------------- GET POINTS FOR RIGHT EYE DRAWING -----------------

                if idx == 362:
                    point_LER = (lm.x * img_w, lm.y * img_h)

                if idx == 374:
                    point_LEB = (lm.x * img_w, lm.y * img_h)

                if idx == 263:
                    point_LEL = (lm.x * img_w, lm.y * img_h)

                if idx == 386:
                    point_LET = (lm.x * img_w, lm.y * img_h)


                # ----------------- GET POINTS FOR IRIS CENTER -----------------
                
                if idx == 468:
                    point_REIC = (lm.x * img_w, lm.y * img_h)

                if idx == 473:
                    point_LEIC = (lm.x * img_w, lm.y * img_h)


    # ---------------------------------- EAR ----------------------------------
                # EAR Right eye
                # ----------------- RIGHT EYE EAR for DROWSINESS DETECTION -----------------
                if idx == 33:
                    p1_right = [lm.x * img_w, lm.y * img_h]

                if idx == 144:
                    p6_right = [lm.x * img_w, lm.y * img_h]

                if idx == 153:
                    p5_right = [lm.x * img_w, lm.y * img_h]

                if idx == 133:
                    p4_right = [lm.x * img_w, lm.y * img_h]

                if idx == 158:
                    p3_right = [lm.x * img_w, lm.y * img_h]
                
                if idx == 160:
                    p2_right = [lm.x * img_w, lm.y * img_h]
                
                # Calculate EAR if all points are feasible
                if p1_right and p2_right and p3_right and p4_right and p5_right and p6_right:
                    all_points_right_eye = True
                    EAR_right = (abs(p2_right[1]-p6_right[1])+abs(p3_right[1]-p5_right[1])) / (2*abs(p1_right[0]-p4_right[0]))
                    if EAR_right > max_EAR_right:
                        max_EAR_right = EAR_right
                    
                    
                # EAR Left eye
                # ----------------- LEFT EYE EAR for DROWSINESS DETECTION -----------------
                if idx == 362:
                    p1_left = [lm.x * img_w, lm.y * img_h]
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                if idx == 380:
                    p6_left = [lm.x * img_w, lm.y * img_h]
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 373:
                    p5_left = [lm.x * img_w, lm.y * img_h]
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 263:
                    p4_left = [lm.x * img_w, lm.y * img_h]
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 387:
                    p3_left = [lm.x * img_w, lm.y * img_h]
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 385:
                    p2_left = [lm.x * img_w, lm.y * img_h]
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                    
                # Calculate EAR if all points are feasible
                if p1_left and p2_left and p3_left and p4_left and p5_left and p6_left:
                    all_points_left_eye = True
                    EAR_left = (abs(p2_left[1]-p6_left[1])+abs(p3_left[1]-p5_left[1])) / (2*abs(p1_left[0]-p4_left[0]))
                    if EAR_left > max_EAR_left:
                        max_EAR_left = EAR_left


                # Drowsiness detection
                # ----------------- DROWSINESS DETECTION -----------------
                if all_points_right_eye and all_points_left_eye:
                    driver_asleep, time_start_drowsy, time_asleep = drowsiness_detection(EAR_right, EAR_left, driver_asleep, time_start_drowsy, time_asleep)     # driver_asleep and time_start_drowsy are global variables
            






    # ---------------------------------- FACE for HEAD GAZE ----------------------------------
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                        cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                    if idx == 33: 
                        right_eye_right_edge_2d = (lm.x * img_w, lm.y * img_h)
                        right_eye_right_edge_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                        cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                    if idx == 61: 
                        mouth_right_2d = (lm.x * img_w, lm.y * img_h)
                        mouth_right_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                        cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                    if idx == 263:
                        left_eye_left_edge_2d = (lm.x * img_w, lm.y * img_h)
                        left_eye_left_edge_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                        cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                    if idx == 291:
                        mouth_left_2d = (lm.x * img_w, lm.y * img_h)
                        mouth_left_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                        cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                    if idx == 199:
                        chin_2d = (lm.x * img_w, lm.y * img_h) 
                        chin_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                        cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                    
                    

                    # Vi må vel kanskje appende alle punkter først, og så gjøre arrayen om til en NUMPY array? fordi numpy array har ikke append funksjonen...

        

                # ----------------- LEFT EYE IRIS for EYE GAZING -----------------
                if idx == 473 or idx == 474 or idx == 475 or idx == 476 or idx == 477: # iris points
                    if idx == 473:
                        left_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        left_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        left_eye_2d.append([x, y])
                        left_eye_3d.append([x, y, lm.z])

                    if idx == 476:
                        left_eye_right_edge_2d = (lm.x * img_w, lm.y * img_h)
                        left_eye_right_edge_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        left_eye_2d.append([x, y])
                        left_eye_3d.append([x, y, lm.z])

                    if idx == 477:
                        left_eye_bottom_2d = (lm.x * img_w, lm.y * img_h)
                        left_eye_bottom_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        left_eye_2d.append([x, y])
                        left_eye_3d.append([x, y, lm.z])

                    if idx == 474:
                        left_eye_left_edge_2d = (lm.x * img_w, lm.y * img_h)
                        left_eye_left_edge_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        left_eye_2d.append([x, y])
                        left_eye_3d.append([x, y, lm.z])

                    if idx == 475:
                        left_eye_top_2d = (lm.x * img_w, lm.y * img_h)
                        left_eye_top_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        left_eye_2d.append([x, y])
                        left_eye_3d.append([x, y, lm.z])
                    
                    
                # ----------------- RIGHT EYE IRIS for EYE GAZING -----------------
                if idx == 468 or idx == 469 or idx == 470 or idx == 471 or idx == 472: # iris points

                    if idx == 468:
                        right_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        right_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        right_eye_2d.append([x, y])
                        right_eye_3d.append([x, y, lm.z])
                    
                    if idx == 471:
                        right_eye_right_edge_2d = (lm.x * img_w, lm.y * img_h)
                        right_eye_right_edge_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        right_eye_2d.append([x, y])
                        right_eye_3d.append([x, y, lm.z])
                    
                    if idx == 472:
                        right_eye_bottom_2d = (lm.x * img_w, lm.y * img_h)
                        right_eye_bottom_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        right_eye_2d.append([x, y])
                        right_eye_3d.append([x, y, lm.z])
                    
                    if idx == 469:
                        right_eye_left_edge_2d = (lm.x * img_w, lm.y * img_h)
                        right_eye_left_edge_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        right_eye_2d.append([x, y])
                        right_eye_3d.append([x, y, lm.z])
                    
                    if idx == 470:
                        right_eye_top_2d = (lm.x * img_w, lm.y * img_h)
                        right_eye_top_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        right_eye_2d.append([x, y])
                        right_eye_3d.append([x, y, lm.z])                    

                    



            # 4.4. - Draw the positions on the frame

            l_eye_width = point_LEL[0] - point_LER[0]

            l_eye_height = point_LEB[1] - point_LET[1]

            l_eye_center = [(point_LEL[0] + point_LER[0])/2 ,(point_LEB[1] + point_LET[1])/2]

            #cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=int(horizontal_threshold * l_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius 

            cv2.circle(image, (int(point_LEIC[0]), int(point_LEIC[1])), radius=3, color=(0, 255, 0), thickness=-1) # Center of iris

            cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye

            #print("Left eye: x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)))

            cv2.putText(image, "Left eye:  x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 



            r_eye_width = point_REL[0] - point_RER[0]

            r_eye_height = point_REB[1] - point_RET[1]

            r_eye_center = [(point_REL[0] + point_RER[0])/2 ,(point_REB[1] + point_RET[1])/2]

            #cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=int(horizontal_threshold * r_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius 

            cv2.circle(image, (int(point_REIC[0]), int(point_REIC[1])), radius=3, color=(0, 0, 255), thickness=-1) # Center of iris

            cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye

            #print("right eye: x = " + str(np.round(point_REIC[0],0)) + " , y = " + str(np.round(point_REIC[1],0)))

            cv2.putText(image, "Right eye: x = " + str(np.round(point_REIC[0],0)) + " , y = " + str(np.round(point_REIC[1],0)), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 

                

            

            # speed reduction (comment out for full speed)

            time.sleep(1/25) # [s]


# ------ CALCULATING FACE AND EYE GAZE ------
        # Do we want to calculate head and eye gaze over again whenever we detect a new point, or is it ok for each frame? If we want to do it for each point, put the calculations inside the for-loop, otherwise, keep it in the if.results.multi_face_landmarks if-statement
        # ----------------- HEAD GAZE -----------------
        # The camera matrix 
        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length   , 0             , img_h / 2],
                                [0              , focal_length  , img_w / 2],
                                [0              , 0             , 1]])
        # The distorsion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Convert to numpy arrays
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        left_eye_2d = np.array(left_eye_2d, dtype=np.float64)
        left_eye_3d = np.array(left_eye_3d, dtype=np.float64)

        right_eye_2d = np.array(right_eye_2d, dtype=np.float64)
        right_eye_3d = np.array(right_eye_3d, dtype=np.float64)
      

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        success_left_eye, rot_vec_left_eye, trans_vec_left_eye = cv2.solvePnP(left_eye_3d, left_eye_2d, cam_matrix, dist_matrix)
        success_right_eye, rot_vec_right_eye, trans_vec_right_eye = cv2.solvePnP(right_eye_3d, right_eye_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        rmat_left_eye, jac_left_eye = cv2.Rodrigues(rot_vec_left_eye)
        rmat_right_eye, jac_right_eye = cv2.Rodrigues(rot_vec_right_eye)

        # Angels
        # rotation about x-axis = pitch, direction of rotation is nodding
        # rotation about y-axis = yaw,   direction of rotation is turing head left or right -> turn head to the right is positive
        # rotation about z-axis = roll,  direction of rotation is tilting head left or right -> tilt to the right is negative

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        angles_left_eye, mtxR_left_eye, mtxQ_left_eye, Qx_left_eye, Qy_left_eye, Qz_left_eye = cv2.RQDecomp3x3(rmat_left_eye)
        angles_right_eye, mtxR_right_eye, mtxQ_right_eye, Qx_right_eye,Qy_right_eye, Qz_right_eye = cv2.RQDecomp3x3(rmat_right_eye)

        # Convert from Euler Angels to degrees
        pitch = angles[0] * 1800
        yaw = -angles[1] * 1800
        roll = 180 + (np.arctan2(point_RER[1] - point_LEL[1], point_RER[0] - point_LEL[0]) * 180 / np.pi)
        if roll > 180:
            roll = roll - 360

        pitch_left_eye = angles_left_eye[0] * 1800
        yaw_left_eye = angles_left_eye[1] * 1800
        pitch_right_eye = angles_right_eye[0] * 1800
        yaq_right_eye = angles_right_eye[1] * 1800

        # Print the angles
        #print("Pitch: " + str(pitch) + " Yaw: " + str(yaw) + " Roll: " + str(roll))
        #print("Pitch left eye: " + str(pitch_left_eye) + " Yaw left eye: " + str(yaw_left_eye))
        #print("Pitch right eye: " + str(pitch_right_eye) + " Yaw right eye: " + str(yaq_right_eye))

        #if head gaze position differs more than +/- 30 degrees with respect to rest-position (0,0,0), print alarm

        end = time.time()

        totalTime = end-start



        if totalTime>0:

            fps = 1 / totalTime

        else:

            fps=0

        

        

        #print("FPS:", fps)



        cv2.putText(image, f'FPS : {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)



        # 4.5 - Show the frame to the user

        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI code sample', image)       
             

# Convert into numpy arrays
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    left_eye_2d = np.array(left_eye_2d, dtype=np.float64)
    left_eye_3d = np.array(left_eye_3d, dtype=np.float64)
    right_eye_2d = np.array(right_eye_2d, dtype=np.float64)
    right_eye_3d = np.array(right_eye_3d, dtype=np.float64)                

    # The camera matrix
    focal_length = 1 * img_w
    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])
    # The distorsion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)


    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    success_left_eye, rot_vec_left_eye, trans_vec_left_eye = cv2.solvePnP(left_eye_3d, left_eye_2d, cam_matrix, dist_matrix)
    success_right_eye, rot_vec_right_eye, trans_vec_right_eye = cv2.solvePnP(right_eye_3d, right_eye_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    rmat_left_eye, jac_left_eye = cv2.Rodrigues(rot_vec_left_eye)
    rmat_right_eye, jac_right_eye = cv2.Rodrigues(rot_vec_right_eye)
    

    if cv2.waitKey(5) & 0xFF == 27:

        break



# 5 - Close properly soruce and eventual log file

cap.release()

#log_file.close()

    

# [EOF]