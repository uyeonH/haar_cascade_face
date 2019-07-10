import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')

face_mask = cv2.imread('data/mask_pororo.png')
h_mask, w_mask = face_mask.shape[:2]
               
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

cap = cv2.VideoCapture("data/mcem0_sa1.mp4")
scaling_factor = 0.5

while True:
   ret, frame = cap.read()
   if not ret :
        break
#    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
   for (x,y,w,h) in face_rects:
        if h > 0 and w > 0:
            #x = int(x + 0.1*w)
            #y = int(y + 0.4*h)
            #w = int(0.8 * w)
            #h = int(0.7 * h)
            #frame 얼굴에 맞춰서 자르기    
            frame_roi = frame[y:y+h, x:x+w]
            #마스크의 크기 조절
            face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
            #그레이마스크    
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            #바탕이 검은색인 마스크를 만듦
            ret, mask = cv2.threshold(gray_mask, 240, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
            frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)
            cv2.imshow('Face mask_inv', mask_inv)
            cv2.imshow(' masked_face', masked_face)
            
   cv2.imshow('Face Detector', frame)
   
   c = cv2.waitKey(1)
   if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
