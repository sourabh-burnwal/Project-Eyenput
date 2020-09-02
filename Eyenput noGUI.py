import cv2
import dlib
from imutils import face_utils
from math import hypot
import numpy as np


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

def nothing(x):
    pass
    
def main():
    font = cv2.FONT_HERSHEY_SIMPLEX
    p = r"C:\Users\soura\Desktop\project eyenput\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    blink = 0
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Live')
    cv2.createTrackbar('threshold', 'Live', 0, 255, nothing)
    while True:
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'Live')
        
        # blink detection
        faces = detector(gray_frame, 0)
        if len(faces) > 0:    
            landmarks = predictor(gray_frame, faces[0])
            face = face_utils.shape_to_np(landmarks)
            left_eye_dots = face[36:42]
            right_eye_dots = face[42:48]
            left_eye_top, left_eye_bot, ratio_left = get_ratio(left_eye_dots)
            right_eye_top, right_eye_bot, ratio_right = get_ratio(right_eye_dots)
            #cv2.line(frame, (left_eye_top[0], left_eye_top[1]), (left_eye_bot[0], left_eye_bot[1]), color=(0,255,0), thickness=1)
            #cv2.line(frame, (face[36,0], face[36,1]), (face[39,0], face[39,1]), color=(0,255,0), thickness=1)
            #cv2.line(frame, (right_eye_top[0], right_eye_top[1]), (right_eye_bot[0], right_eye_bot[1]), color=(0,255,0), thickness=1)
            #cv2.line(frame, (face[42,0], face[42,1]), (face[45,0], face[45,1]), color=(0,255,0), thickness=1)
            aggr = (ratio_left+ratio_right) / 2
            if aggr >= 18.755:
                blink = blink+1
                cv2.putText(frame, 'Blinking'.format(blink), (50,100), font, 1, (255,0,0))
            
        # gaze detection
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
        
        left_eye_min_x, left_eye_min_y, left_eye_max_x, left_eye_max_y = get_bounds(left_eye_region)
        right_eye_min_x, right_eye_min_y, right_eye_max_x, right_eye_max_y = get_bounds(right_eye_region)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 1)
        cv2.fillPoly(mask, [left_eye_region], 255)
        cv2.polylines(mask, [right_eye_region], True, 255, 1)
        cv2.fillPoly(mask, [right_eye_region], 255)

        eyes_mask = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        left_eye = eyes_mask[left_eye_min_y:left_eye_max_y, left_eye_min_x:left_eye_max_x]
        left_eye_resized = cv2.resize(left_eye, None, fx = 5, fy = 5)
        _, left_eye_mod = cv2.threshold(left_eye_resized, threshold, 255, cv2.THRESH_BINARY)
        left_eye_mod = cv2.erode(left_eye_mod, None, iterations=2)
        left_eye_mod = cv2.dilate(left_eye_mod, None, iterations=3)
        left_eye_mod = cv2.medianBlur(left_eye_mod, 5)

        cv2.polylines(frame, [left_eye_region], True, (0,255,0), 1)
        cv2.polylines(frame, [right_eye_region], True, (0,255,0), 1)    


        cv2.imshow("Live", frame)
        cv2.imshow("Left Eye", left_eye_mod)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()

