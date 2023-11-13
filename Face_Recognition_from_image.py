import face_recognition
import cv2
import numpy as np

image= cv2.imread("99.jpg")

face1= face_recognition.load_image_file("1.png")
face1_encoding= face_recognition.face_encodings(face1)[0]

face2= face_recognition.load_image_file("2.jpg")
face2_encoding= face_recognition.face_encodings(face2)[0]

face3= face_recognition.load_image_file("3.jpg")
face3_encoding= face_recognition.face_encodings(face3)[0]

face4= face_recognition.load_image_file("4.jpg")
face4_encoding= face_recognition.face_encodings(face4)[0]

face5= face_recognition.load_image_file("5.jpg")
face5_encoding= face_recognition.face_encodings(face5)[0]

known_face_encodings= [
    face1_encoding,
    face2_encoding,
    face3_encoding,
    face4_encoding,
    face5_encoding
]

known_face_names=[
    "Jake Peralta",
    "Amy Santiago",
    "Terry Jeffords",
    "Charles Boyle",
    "Raymond Holt"
]

face_locations= []
face_encodings= []
face_names= []

rgb_image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
face_locations= face_recognition.face_locations(rgb_image)
face_encodings= face_recognition.face_encodings(rgb_image, face_locations)
face_names= []
for face_encoding in face_encodings:
    matches= face_recognition.compare_faces(known_face_encodings, face_encoding)
    name= "Unknown"

    face_distances= face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index= np.argmin(face_distances)
    if matches[best_match_index]:
        name= known_face_names[best_match_index]
    face_names.append(name)

for (top, right, bottom, left), name in zip(face_locations, face_names):
    top *= 1
    right *= 1
    bottom *= 1
    left*= 1

    cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),1)
    cv2.rectangle(image,(left,bottom-20),(right,bottom),(255,0,0),cv2.FILLED)
    font=cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image,name,(left+6,bottom-6),font,0.5,(255,255,255),1) 

cv2.imshow("Faces Found", image)
cv2.waitKey(0)
