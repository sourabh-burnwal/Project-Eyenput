import cv2
from imutils import face_utils
from math import hypot
import numpy as np
from scipy.spatial import distance


def midpoint(p1, p2):
    return [(p1[0]+p2[0])//2, (p1[1]+p2[1])//2]

def get_ratio(eyes_dots):
    eye_hor_len = hypot((eyes_dots[0,0]-eyes_dots[3,0]), eyes_dots[0,1]-eyes_dots[3,1])
    eye_top = midpoint(eyes_dots[1,:], eyes_dots[2,:])
    eye_bot = midpoint(eyes_dots[4,:], eyes_dots[5,:])
    eye_ver_len = hypot((eye_top[0]-eye_bot[0]), eye_top[1], eye_bot[1])
    ratio = eye_ver_len / eye_hor_len
    return eye_top, eye_bot, ratio

def get_bounds(region):
    min_x = np.min(region[:,0])
    max_x = np.max(region[:,0])
    min_y = np.min(region[:,1])
    max_y = np.max(region[:,1])
    return min_x, min_y, max_x, max_y

def get_landmarks(frame, detector, predictor):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame, 0)
    if len(faces) > 0:    
        landmarks = predictor(gray_frame, faces[0])
        face = face_utils.shape_to_np(landmarks)
        return face, gray_frame, landmarks


def eye_aspect_ratio(eyes_dots):
    A = distance.euclidean(eyes_dots[1,:], eyes_dots[5,:])
    B = distance.euclidean(eyes_dots[2,:], eyes_dots[4,:])
    C = distance.euclidean(eyes_dots[0,:], eyes_dots[3,:])
    ear = (A + B) / (2.0 * C)
    return ear

def blink_counter(face, blink, counter):
    ear_thres = 0.358
    frames_per_blink = 5

    left_eye_dots = face[36:42]
    right_eye_dots = face[42:48]
    left_ear = eye_aspect_ratio(left_eye_dots)
    right_ear = eye_aspect_ratio(right_eye_dots)
    aggr_ear = left_ear + right_ear / 2
    if aggr_ear < ear_thres:
        counter += 1
    if counter >= frames_per_blink:
        blink += 1
        counter = 0
    return blink, counter

def eye_prepare(eyes_mask, region, threshold):
    eye_min_x, eye_min_y, eye_max_x, eye_max_y = get_bounds(region)
    
    eye = eyes_mask[eye_min_y:eye_max_y, eye_min_x:eye_max_x]
    eye_resized = cv2.resize(eye, None, fx = 12, fy = 12)
    _, eye_mod = cv2.threshold(eye_resized, threshold, 255, cv2.THRESH_BINARY)
    eye_mod = cv2.erode(eye_mod, None, iterations=2)
    eye_mod = cv2.dilate(eye_mod, None, iterations=4)
    eye_mod = cv2.medianBlur(eye_mod, 5)
    return eye_mod


def eyes_preproc(landmarks, frame, gray_frame, left_eye_thresh, right_eye_thresh):
    left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)])

    right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                (landmarks.part(43).x, landmarks.part(43).y),
                                (landmarks.part(44).x, landmarks.part(44).y),
                                (landmarks.part(45).x, landmarks.part(45).y),
                                (landmarks.part(46).x, landmarks.part(46).y),
                                (landmarks.part(47).x, landmarks.part(47).y)])
    
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 1)
    cv2.fillPoly(mask, [left_eye_region], 255)
    cv2.polylines(mask, [right_eye_region], True, 255, 1)
    cv2.fillPoly(mask, [right_eye_region], 255)

    eyes_mask = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
    left_eye_mod = eye_prepare(eyes_mask, left_eye_region, left_eye_thresh)
    right_eye_mod = eye_prepare(eyes_mask, right_eye_region, right_eye_thresh)
    return left_eye_mod, right_eye_mod
    
def detect_blob(eye, detector):
    keypoints = detector.detect(eye)
    return keypoints
