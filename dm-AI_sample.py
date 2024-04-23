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

    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    
    # 4.3 - Get the landmark coordinates

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            for idx, lm in enumerate(face_landmarks.landmark):


                # Eye Gaze (Iris Tracking)

                # Left eye indices list

                #LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

                # Right eye indices list

                #RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
     
                #LEFT_IRIS = [473, 474, 475, 476, 477]

                #RIGHT_IRIS = [468, 469, 470, 471, 472]

                if idx == 33:

                    point_RER = (lm.x * img_w, lm.y * img_h)

                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 145:

                    point_REB = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                
                if idx == 144:

                    point_REBL = (lm.x * img_w, lm.y * img_h)

                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                
                if idx == 153:

                    point_REBR = (lm.x * img_w, lm.y * img_h)

                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
        

                if idx == 133:

                    point_REL = (lm.x * img_w, lm.y * img_h)

                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 159:

                    point_RET = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                

                if idx == 158:
                    point_RETR = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 160:
                    point_RETL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)


                if idx == 362:

                    point_LER = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 374:

                    point_LEB = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 263:

                    point_LEL = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 386:

                    point_LET = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)

                if idx == 468:

                    point_REIC = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 255, 0), thickness=-1)                    

                if idx == 469:

                    point_469 = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 470:

                    point_470 = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 471:

                    point_471 = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 472:

                    point_472 = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 0), thickness=-1)

                if idx == 473:

                    point_LEIC = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 255, 255), thickness=-1)

                if idx == 474:

                    point_474 = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 475:

                    point_475 = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 476:

                    point_476 = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)

                if idx == 477:

                    point_477 = (lm.x * img_w, lm.y * img_h)

                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(255, 0, 0), thickness=-1)



                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:

                    if idx == 1:

                        nose_2d = (lm.x * img_w, lm.y * img_h)

                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)



                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                   
                    # Get the 2D Coordinates
                    face_2d.append([x, y])
                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])


                #LEFT_IRIS = [473, 474, 475, 476, 477]

                if idx == 473 or idx == 362 or idx == 374 or idx == 263 or idx == 386: # iris points

                #if idx == 473 or idx == 474 or idx == 475 or idx == 476 or idx == 477: # eye border

                    if idx == 473:

                        left_pupil_2d = (lm.x * img_w, lm.y * img_h)

                        left_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    left_eye_2d.append([x, y])
                    left_eye_3d.append([x, y, lm.z])


                #RIGHT_IRIS = [468, 469, 470, 471, 472]

                if idx == 468 or idx == 33 or idx == 145 or idx == 133 or idx == 159: # iris points

                # if idx == 468 or idx == 469 or idx == 470 or idx == 471 or idx == 472: # eye border

                    if idx == 468:

                        right_pupil_2d = (lm.x * img_w, lm.y * img_h)

                        right_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    right_eye_2d.append([x, y])
                    right_eye_3d.append([x, y, lm.z])

                    
                # EAR 
                #RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]      
                #if idx == 33 or idx == 144 or idx == 153 or idx == 133 or idx == 158 or idx == 160:
                if idx == 33:
                    p1 = [lm.x * img_w, lm.y * img_h]

                if idx == 144:
                    p6 = [lm.x * img_w, lm.y * img_h]

                if idx == 153:
                    p5 = [lm.x * img_w, lm.y * img_h]

                if idx == 133:
                    p4 = [lm.x * img_w, lm.y * img_h]

                if idx == 158:
                    p3 = [lm.x * img_w, lm.y * img_h]
                
                if idx == 160:
                    p2 = [lm.x * img_w, lm.y * img_h]

                if p1 and p2 and p3 and p4 and p5 and p6:
                    EAR_right = (abs(p2[1]-p6[1])+abs(p3[1]-p5[1])) / (2*abs(p1[0]-p4[0]))
                    print(EAR_right)
                    #print(type(EAR_right))
                    #cv2.putText(image, "EAR_right:  x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                
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

