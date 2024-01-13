# Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
# face_utils for basic operations of conversion
from imutils import face_utils
import time
import pygame
import smtplib

path="shape_predictor_68_face_landmarks.dat"
predictor=dlib.shape_predictor(path)
detector=dlib.get_frontal_face_detector()


# creates SMTP session
s = smtplib.SMTP('smtp.gmail.com', 587)
# start TLS for security
s.starttls()
# Authentication
s.login("omkinge93@gmail.com", "mlpwhfatemtwzrij") # login with your mail and password
# message to be sent
message = "Your Driver was sleeping "

def get_landmarks(im):
    rects=detector(im,1)#image and no.of rectangles to be drawn
    if len(rects)>1:
        print("Toomanyfaces")
        return np.matrix([0])
    if len(rects)==0:
        print("Toofewfaces")
        return np.matrix([0])
    return np.matrix([[p.x,p.y] for p in predictor(im,rects[0]).parts()])   
def place_landmarks(im,landmarks):
    im=im.copy()
    for idx,point in enumerate(landmarks):
        pos=(point[0,0],point[0,1])
        cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.3,color=(0,255,255))
        cv2.circle(im,pos,3,color=(0,255,255))
    return im 
def upper_lip(landmarks):
    top_lip=[]
    for i in range(50,53):
        top_lip.append(landmarks[i])
    for j in range(61,64):
        top_lip.append(landmarks[j])
    top_lip_point=(np.squeeze(np.asarray(top_lip)))
    top_mean=np.mean(top_lip_point,axis=0)
    return int(top_mean[1])
    
        
def low_lip(landmarks):
    lower_lip=[]
    for i in range(65,68):
        lower_lip.append(landmarks[i])
    for j in range(56,59):
        lower_lip.append(landmarks[j])
    lower_lip_point=(np.squeeze(np.asarray(lower_lip)))
    lower_mean=np.mean(lower_lip_point,axis=0)
    
        
    return int(lower_mean[1])
               
def decision(image):
    landmarks=get_landmarks(image)
    if(landmarks.all()==[0]):
        return -10#Dummy value to prevent error
    top_lip=upper_lip(landmarks)
    lower_lip=low_lip(landmarks)
    distance=abs(top_lip-lower_lip)
    return distance

pygame.mixer.init()
pygame.mixer.music.load('police-operation-siren-144229.wav')

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)
yawns=0
    
# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if (ratio > 0.25):
        return 2
    elif (ratio > 0.21 and ratio <= 0.25):
        return 1
    else:
        return 0


while True:
    ret,frame=cap.read()
    if(ret==True):
        
        landmarks=get_landmarks(frame)
        if(landmarks.all()!=[0]):
            l1=[]
            for k in range(48,60):
                l1.append(landmarks[k])
            l2=np.asarray(l1)
            lips=cv2.convexHull(l2)
            cv2.drawContours(frame, [lips], -1, (0, 255, 0), 1)
        
        distance=decision(frame)
        if(distance>21):    #Use distance according to your convenience
            yawns=yawns+1
            
        cv2.putText(frame,"Yawn Count: "+str(yawns),(50,100),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(255,0,0))
        cv2.imshow("Subject Yawn Count",frame)
        if cv2.waitKey(1)==13:
            break
    else:
        continue
    if(yawns>=10):
        pygame.mixer.music.play(-1)
        time.sleep(5)
        pygame.mixer.music.stop()
        yawns = 0
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fra = cv2.rectangle(frame, (0, 0), (230, 70), (0, 0, 0), 10, None, None)
    faces = detector(gray)
    # detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_copy = frame.copy()
        cv2.rectangle(face_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Now judge what to do for the eye blinks
        if (left_blink == 0 or right_blink == 0):
            sleep += 1
            drowsy = 0
            active = 0
            if (sleep > 6):
                pygame.mixer.music.play(-1)
                status = "SLEEPING        "#7
                 # sending the mail
                s.sendmail("patilshreyash8555@gmail.com", "shreyaspatel434@gmail.com", message)
                # terminating the session
                #s.quit()
                color = (255, 0, 0)


        elif (left_blink == 1 or right_blink == 1):
            sleep = 0
            active = 0
            drowsy += 1
            if (drowsy > 6):
                pygame.mixer.music.stop()
                status = "DROWSY!!          "#8
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if (active > 6):
                pygame.mixer.music.stop()
                status = " ACTIVE          "#8
                color = (0, 200, 0)
        stat = "Facelandmarks"
        cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(face_copy, stat, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_copy, (x, y), 1, (255, 255, 255), -1)

    #hor = np.hstack(frame,face_copy)
    #cv2.imshow("Result",hor)
    cv2.imshow("Drowsiness Status", frame)
    #cv2.imshow("Face Landmarks", face_copy)
    key = cv2.waitKey(1)
    if key == 27:
        break