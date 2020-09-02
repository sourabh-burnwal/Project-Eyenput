from imutils import face_utils
import dlib
import cv2
from numpy import linalg as lg
from time import sleep

#using downloaded landmark predictor used by dlib
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
#initialising blink count
blink = 0

while True:
    # Getting out image by webcam 
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        #finding the distance between the landmark points of eyes in dlib
        #on blinking the norm will be least
        if lg.norm(shape[37]-shape[41]) < 7.0:
            blink += 1
            print("blink: {}".format(blink))
            sleep(0.02)
        
        """"
        if lg.norm(shape[37]-shape[41]) < 7.0:
            blink += 0.5
            if blink*10 % 10 == 0.0:
                print("blink: {}".format(blink))
                """
        # Draw on our image, all the found cordinate points (x,y) 
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        #print(shape[36:47])
    # Show the image
    cv2.imshow("Output", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
