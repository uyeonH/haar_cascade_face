import cv2
import numpy as np
import pafy

nose_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
	raise IOError('Unable to load the mouth cascade classifier xml file')
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')
if eye_cascade.empty():
	raise IOError('Unable to load the eye cascade classifier xml file')
if nose_cascade.empty():
	raise IOError('Unable to load the nose cascade classifier xml file')

ds_factor = 1.5
src1 = cv2.imread('./data/face.png')
cv2.imshow('Face Detector', src1)
frame=src1
    
#frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
#여러개의 정보가 들어갈 수 있음, 
for (x,y,w,h) in face_rects:
    #cv2.rectangle(frame, (x,y), (x+w,y+int(h*0.7)), (255,255,0), 3)
    cv2.rectangle(frame,(x,int(y-h*0.2)),(x+w,int(y+h*1.1)),(255,0,0),3)
    #print(x,int(y-h*0.2),x+w,int(y+h*1.1))
    #face=frame[x:int(x+w*0.6),int(y-h*0.2):int(y+h*2)]
    face=frame[29:136,26:165]
    #cv2.imwrite('./face/face.png',face)
    #cv2.imshow('face',face)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
for (x_eye,y_eye,w_eye,h_eye) in eyes:
    center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
    radius = int(0.3 * (w_eye + h_eye))
    color = (160, 255, 160)
    thickness = 3
    cv2.circle(roi_color, center, radius, color, thickness)
    
for (x,y,w,h) in mouth_rects:
    y = int(y - 0.15*h)
    cv2.rectangle(frame, (x,int(y+h*0.3)), (int(x+w*1.4),int(y+h*0.7)), (0,255,0), 3)
    lip=frame[x:int(x+w*1.4),int(y+h*0.3):int(y+h*0.7)]
    #cv2.imwrite('./face/lip.png',lip)
    #cv2.imshow('lip',lip)
    #break

for (x,y,w,h) in nose_rects:
    cv2.rectangle(frame, (int(x+w*0.2),int(y-0.7*h)), (x+w,y+h), (0,220,220),3)
    #nose=frame[x+w*0.2:x+w,y-0.7*h:y+h]
    #cv2.imwrite('./face/nose.png',nose)
    #cv2.imshow('nose',nose)
    #break


cv2.imshow('Face Detector Rect', frame)
c = cv2.waitKey()
cv2.destroyAllWindows()
