import cv2
import numpy as np
#classifier 계층기?

face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')

if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file') #path 잘 설정

cap = cv2.VideoCapture('./data/mcem0_sa1.mp4')
scaling_factor = 0.5

while True:
    ret, frame = cap.read()

    if not ret:
            break
    #frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    #여러개의 정보가 들어갈 수 있음, 
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    cv2.imshow('Face Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
