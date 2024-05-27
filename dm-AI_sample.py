import cv2
import mediapipe as mp
import numpy as np 
import math
import time


def drowsiness_detection(EAR_right, EAR_left, driver_asleep, time_start_drowsy, time_asleep):

    EAR_treshold = 0.2
    left_eye_closed = False
    right_eye_closed = False

    if EAR_right < EAR_treshold:
        right_eye_closed = True
  
    if EAR_left < EAR_treshold:
        left_eye_closed = True

    if right_eye_closed and left_eye_closed:
        if driver_asleep == False:
            print("Possible drowsiness detected\n")
            cv2.putText(image, "Possible drowsiness detected", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            time_start_drowsy = time.time()
            driver_asleep = True
        elif driver_asleep == True:
            time_asleep = time.time() - time_start_drowsy

    if not right_eye_closed and not left_eye_closed:
        driver_asleep = False
        time_asleep = 0

    if time_asleep > 10:
        print("Eyes closed for too long")      # !!! ALARM !!!
        cv2.putText(image, "Eyes closed!!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return driver_asleep, time_start_drowsy, time_asleep


# Function to calculate the yaw angle between the Iris Center and the Center of the Eye
def yaw_angle_between_points(point_center, point_iris, point_ref):     
    sign = 1                                
    if point_iris[0] < point_center[0]:         # negative angle if iris is to the left of the center of the eye (looking to the left)
        sign = -1  

    # Calculate the vector between the reference point and the iris center and the center of the eye
    vec_ref_iris   = [point_iris[0] - point_ref[0], point_iris[1] - point_ref[1]]
    vec_ref_center = [point_center[0] - point_ref[0], point_center[1] - point_ref[1]]

    # Dot product
    dot_product = np.dot(vec_ref_iris, vec_ref_center)
    norm_ref_iris = np.linalg.norm(vec_ref_iris)
    norm_ref_center = np.linalg.norm(vec_ref_center)

    cos_theta = dot_product / (norm_ref_iris * norm_ref_center)
    
    angle_in_radians = np.arccos(cos_theta)
    angle_in_degrees = math.degrees(angle_in_radians)

    return angle_in_degrees*sign


# Function to calculate the pitch angle between the Iris Center and the Center of the Eye
def pitch_angle_between_points(point_center, point_iris, point_ref):     
    sign = 1                                
    if point_iris[1] > point_center[1]:             # negative angle if iris is below the center of the eye (looking down)
        sign = -1  

    # Calculate the vector between the reference point and the iris center and the center of the eye
    vec_ref_iris   = [point_iris[0] - point_ref[0], point_iris[1] - point_ref[1]]
    vec_ref_center = [point_center[0] - point_ref[0], point_center[1] - point_ref[1]]

    # Dot product
    dot_product = np.dot(vec_ref_iris, vec_ref_center)
    norm_ref_iris = np.linalg.norm(vec_ref_iris)
    norm_ref_center = np.linalg.norm(vec_ref_center)

    cos_theta = dot_product / (norm_ref_iris * norm_ref_center)
    
    angle_in_radians = np.arccos(cos_theta)
    angle_in_degrees = math.degrees(angle_in_radians)

    return angle_in_degrees*sign



# Settings for MediaPipe Face Mesh

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


# ----- OPEN VIDEO CAPTURE -----
cap = cv2.VideoCapture(0) # Local webcam (index start from 0)
driver_asleep = False
time_start_drowsy = 0
time_asleep = 0
all_points_left_eye = False
all_points_right_eye = False


while cap.isOpened(): 

    # Get the new frame
    success, image = cap.read() 
    
    start = time.time()


    if image is None:
        break

    image.flags.writeable = False

    
    # Run MediaPipe on the frame
    results = face_mesh.process(image)



    # To improve performance
    image.flags.writeable = True


    img_h, img_w, img_c = image.shape

    # making new arrays for each frame in the "video", calcluating the points for each frame, and then calculating the gaze for each frame. If we want to calculate the gaze more often, we have to watch when we convert the arrays to numpy arrays because numpy arrays do not have the "append" function.
    face_2d = []
    face_3d = []

    # RIGHT EYE
    point_RER = [] # Right Eye Right
    point_REB = [] # Right Eye Bottom
    point_REL = [] # Right Eye Left
    point_RET = [] # Right Eye Top

    # LEFT EYE
    point_LER = [] # Left Eye Right
    point_LEB = [] # Left Eye Bottom
    point_LEL = [] # Left Eye Left
    point_LET = [] # Left Eye Top

    # IRISES
    point_REIC = [] # Right Eye Iris Center
    point_LEIC = [] # Left Eye Iris Center

    # EAR POINTS
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
    
    # Get the coordinates of the facial landmarks
    if results.multi_face_landmarks:                            # Check if any face is detected

        for face in results.multi_face_landmarks:               # Iterate through all detected faces 

            for idx, lm in enumerate(face.landmark):            # Iterate through all landmarks in face. idx = point number, lm = point coordinate 

                # ----------------- SOME IMPORTANT POINTS -----------------

                #LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
                #LEFT_EYE_BORDER=[ 473, 362, 374, 263, 386 ]

                #RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
                #RIGHT_EYE_BORDER=[ 468, 33, 145, 133, 159 ]

                #LEFT_IRIS  = [ 473, 474, 475, 476, 477 ] 
                #RIGHT_IRIS = [ 468, 469, 470, 471, 472 ]




                # ----------------- RIGHT EYE -----------------
                # Draw eye and later used for calculating center of eye
                if idx == 33:
                    point_RER = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 145:
                    point_REB = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
        
                if idx == 133:
                    point_REL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 159:
                    point_RET = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)


                # ----------------- LEFT EYE -----------------
                # Draw eye and later used for calculating center of eye
                if idx == 362:
                    point_LER = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 374:
                    point_LEB = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 263:
                    point_LEL = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)

                if idx == 386:
                    point_LET = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)


                # ----------------- GET POINTS FOR IRIS CENTER -----------------
                if idx == 468:
                    point_REIC = (lm.x * img_w, lm.y * img_h)

                if idx == 473:
                    point_LEIC = (lm.x * img_w, lm.y * img_h)



    # -------------------------------------------------------------------------
    # ------------------- EAR for DROWSINESS DETECTION ------------------------
    # -------------------------------------------------------------------------
                # ----------------- RIGHT EYE -----------------
                if idx == 33:
                    p1_right = [lm.x * img_w, lm.y * img_h]

                if idx == 160:
                    p2_right = [lm.x * img_w, lm.y * img_h]

                if idx == 158:
                    p3_right = [lm.x * img_w, lm.y * img_h]

                if idx == 133:
                    p4_right = [lm.x * img_w, lm.y * img_h]

                if idx == 153:
                    p5_right = [lm.x * img_w, lm.y * img_h]

                if idx == 144:
                    p6_right = [lm.x * img_w, lm.y * img_h]

        
                # Calculate EAR if all points are feasible
                if p1_right and p2_right and p3_right and p4_right and p5_right and p6_right:
                    all_points_right_eye = True
                    EAR_right = (abs(p2_right[1]-p6_right[1])+abs(p3_right[1]-p5_right[1])) / (2*abs(p1_right[0]-p4_right[0]))
        

                # ----------------- LEFT EYE  -----------------
                if idx == 362:
                    p1_left = [lm.x * img_w, lm.y * img_h]

                if idx == 385:
                    p2_left = [lm.x * img_w, lm.y * img_h]

                if idx == 387:
                    p3_left = [lm.x * img_w, lm.y * img_h]

                if idx == 263:
                    p4_left = [lm.x * img_w, lm.y * img_h]

                if idx == 373:
                    p5_left = [lm.x * img_w, lm.y * img_h]

                if idx == 380:
                    p6_left = [lm.x * img_w, lm.y * img_h]


                # Calculate EAR if all points are feasible
                if p1_left and p2_left and p3_left and p4_left and p5_left and p6_left:
                    all_points_left_eye = True
                    EAR_left = (abs(p2_left[1]-p6_left[1])+abs(p3_left[1]-p5_left[1])) / (2*abs(p1_left[0]-p4_left[0]))
                
                
                # ----------------- DROWSINESS DETECTION -----------------
                if all_points_right_eye and all_points_left_eye:
                    driver_asleep, time_start_drowsy, time_asleep = drowsiness_detection(EAR_right, EAR_left, driver_asleep, time_start_drowsy, time_asleep)     # driver_asleep and time_start_drowsy are global variables
            



    # ----------------------------------------------------------------------------------------
    # ---------------------------------- FACIAL ORIENTATION ----------------------------------
    # ----------------------------------------------------------------------------------------
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
                    


            # Draw Iris center and Center of Eye for both eyes
            # ----- LEFT EYE -----
            l_eye_center = [(point_LEL[0] + point_LER[0])/2 ,(point_LEB[1] + point_LET[1])/2]


            cv2.circle(image, (int(point_LEIC[0]), int(point_LEIC[1])), radius=3, color=(0, 255, 0), thickness=-1) # Center of iris
            cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye

            cv2.putText(image, "Left eye:  x = " + str(np.round(point_LEIC[0],0)) + " , y = " + str(np.round(point_LEIC[1],0)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 


            # ----- RIGHT EYE -----
            r_eye_center = [(point_REL[0] + point_RER[0])/2 ,(point_REB[1] + point_RET[1])/2]

            cv2.circle(image, (int(point_REIC[0]), int(point_REIC[1])), radius=3, color=(0, 0, 255), thickness=-1) # Center of iris
            cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye
            
            cv2.putText(image, "Right eye: x = " + str(np.round(point_REIC[0],0)) + " , y = " + str(np.round(point_REIC[1],0)), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 

            
            time.sleep(1/25) # [s]



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
      

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Angels
        # pitch -> nodding: negative down, positive up
        # yaw -> turn: negative left, positive right
        # roll -> tilting: negative left, positive right

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        # Convert from Euler Angels to degrees
        pitch = angles[0] * 1800
        yaw = -angles[1] * 1800
        roll = 180 + (np.arctan2(point_RER[1] - point_LEL[1], point_RER[0] - point_LEL[0]) * 180 / np.pi)
        if roll > 180:
            roll = roll - 360

        
    # ----------------- EYE GAZE -----------------
        # Finding the center of the eye
        point_LEC = l_eye_center
        point_REC = r_eye_center
        
        # Calculating angle between Iris Center and Center of Eye
        right_eye_yaw_angle = yaw_angle_between_points(point_REIC, point_REC, point_RET)
        right_eye_pitch_angle = pitch_angle_between_points(point_REIC, point_REC, point_RER)

        left_eye_yaw_angle = yaw_angle_between_points(point_LEIC, point_LEC, point_LET)
        left_eye_pitch_angle = pitch_angle_between_points(point_LEIC, point_LEC, point_LER)

        total_pitch = pitch + right_eye_pitch_angle/2 + left_eye_pitch_angle/2
        total_yaw = yaw + right_eye_yaw_angle/2 + left_eye_yaw_angle/2
        total_roll = roll

        if abs(total_pitch) > 30 or abs(total_yaw) > 30 or abs(total_roll) > 30:
            print("!!ALARM!! Diver is distracted")
            cv2.putText(image, "Driver asleep", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        end = time.time()

        totalTime = end-start



        if totalTime>0:

            fps = 1 / totalTime

        else:

            fps=0



        cv2.putText(image, f'FPS : {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI code sample', image)       
             
    

    if cv2.waitKey(5) & 0xFF == 27:

        break



cap.release()
