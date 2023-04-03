import cv2

face_cascade = cv2.CascadeClassifier()

face_cascade.load("haarcascade_frontal.xml")

frame = cv2.imread("../../data/test10.jpg")

frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame_gray = cv2.equalizeHist(frame_gray)
#-- Detect faces
faces = face_cascade.detectMultiScale(frame_gray)

for (x,y,w,h) in faces:
    frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    
cv2.imwrite("test.jpg", frame)