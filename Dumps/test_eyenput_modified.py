import cv2
import numpy as np
import imutils
from imutils import face_utils
import dlib
from numpy import linalg as lg
from time import sleep
import pyautogui as gui
#from curtsies import Input


#function to detect eyes
def detect_eyes(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #using haar cascades to detect eyes
    eyes = classifier.detectMultiScale(gray_frame, 1.3, 5)
    width = np.size(img, 1)
    height = np.size(img, 0)
    left_eye = None
    right_eye = None
    left_eye_coords = []
    right_eye_coords = []
    #filtering out the wrong detections
    #as the eyes will be in upper half of the face
    for (x,y,w,h) in eyes:
        
        eye_center = x + w / 2
        if eye_center < width/2:
            left_eye_coords.append([x,y,w,h])
            left_eye = img[y:y+h, x:x+w]
        else:
            right_eye = img[y:y+h, x:x+w]
            right_eye_coords.append([x,y,w,h])
    return left_eye, right_eye, left_eye_coords, right_eye_coords

#function to detect faces in the image
def detect_faces(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #using haar cascades to detect faces
    face_coords = classifier.detectMultiScale(gray_frame, 1.3, 5)
    face_loc = [[0]]
    #filtering out the biggest detected face in the image
    if len(face_coords) > 1:
        biggest = (0,0,0,0)
        for i in face_coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(face_coords) == 1:
        biggest = face_coords
    else:
        return None
    for (x,y,w,h) in biggest:
        frame = img[y:y+h, x:x+w]
        face_loc.append([x,y,w,h])
    return frame, face_loc

#function to extract only the eyes
def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height/4)
    img = img[eyebrow_h:height, 0:width]
    return img

#function to extract blob in the eyes
def eye_locator(img, threshold):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #converting the image into binary
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    
    #post-processing the image for better blob-detection
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    cX = []
    cY = []
    
    #detecting the contours
    cnts = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #print(cnts)
    for i in range(len(cnts)):
    # compute the center of the contour
        M = cv2.moments(cnts[i])
        cX.append(int(M["m10"] / M["m00"]))
        cY.append(int(M["m01"] / M["m00"]))
    return cnts, cX, cY

#will help keeping the trackbar consistent
def nothing(x):
    pass

def blink_counter(frame, detector, predictor, blink):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
            #print("blink: {}".format(blink))
            sleep(0.02)
        #to draw locator points on the face
        #for (x, y) in shape:
            #cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
    return blink

def control_screen(cX, cY):
    _x = cX - 218
    _y = cY - 216
    x = int(13.66 * _x)
    y = int(13.24 * _y)
    gui.moveTo(1366-x, 768-y, 0.3)
    #print("({},{})".format(x,y))

#main function
def main():
    gui.FAILSAFE = False
    #haar cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    #pre-requisite for dlib detector and predictors
    p = "project eyenput\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    
    #initializing the blink value
    blink = 0
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    #creating the trackbar
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    flag = 0
    while True:
        _, frame = cap.read()
        
        #finding the blink count
        blink = blink_counter(frame, detector, predictor, blink)
        #detecting faces
        face_frame, face_loc = detect_faces(frame, face_cascade)
        x_face, y_face, w_face, h_face = face_loc[1]
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            #for eye in eyes[:2]:
            for i in range(0,2):
                if eyes[i] is not None:
                    
                    #taking threshold as input from the user
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eyes[i])
                    #locating and processing the eye-part
                    c, cX, cY = eye_locator(eye, threshold)
                    temp = eyes[int(len(eyes)/2)+i]
                    x,y,w,h = temp[0]
                    h_new = int(h / 4)
                    y_new = y + (h - h_new)
                    if len(c) >= 2:
                        for i in range(1,len(c)):
                            #drawing the blob keypoints onto the eyes
                            cv2.drawContours(eye, [c[i]], -1, (0, 0, 255), 1)
                            cv2.circle(eye, (cX[i], cY[i]), 2, (0, 255, 0), -1)
                            cX_wind = cX[i] + x + x_face
                            cY_wind = cY[i] + y_new + y_face
                        if flag == 0:
                            if input() == "t":
                                flag = 1
                        elif flag == 1 and i == 1:
                            control_screen(cX_wind, cY_wind)
        flipped = cv2.flip(frame,1)
        #flipped = cv2.rectangle(flipped,(320,217),(420,275),(255,0,0),2)
        flipped = cv2.rectangle(flipped,(219,217),(319,275),(255,0,0),2)
        cv2.imshow('image', flipped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#driver function
if __name__  == '__main__':
    main()
