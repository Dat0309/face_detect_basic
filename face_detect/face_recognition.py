import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier(r'C:\Users\ADMIN\Desktop\haar_face.xml')

people= ['Hai', 'Trong Dat', 'stranger', 'Tri']
# feature = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\ADMIN\Desktop\face\stranger\meme.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Person',gray)

#Nhận diện khuôn mặt
faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect:
    face_roi = gray[y:y+h,x:x+h]

    label, confidence = face_recognizer.predict(face_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_SIMPLEX, 1.0,(255,0,0), thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=2)

cv.imshow('Nhận diện',img)
cv.waitKey(0)