import cv2
import numpy as np
import pafy
nose_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
	raise IOError('Unable to load the mouth cascade classifier xml file')
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

if eye_cascade.empty():
	raise IOError('Unable to load the eye cascade classifier xml file')

if nose_cascade.empty():
	raise IOError('Unable to load the nose cascade classifier xml file')
url='https://www.youtube.com/watch?v=SXCJpaJ0sFo'
video=pafy.new(url)
best=video.getbest(preftype='webm')
cap = cv2.VideoCapture(best.url)
ds_factor = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break
#    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    #여러개의 정보가 들어갈 수 있음, 
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 3)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)    
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
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        break
    for (x,y,w,h) in nose_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,220,220), 3)
        
        break

    cv2.imshow('Nose Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
