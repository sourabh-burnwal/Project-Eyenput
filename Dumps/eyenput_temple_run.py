import cv2
import numpy as np
import pyautogui as pag
from time import sleep


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Live')
    _, first_frame = cap.read()
    first_frame = cv2.flip(first_frame, 1)

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (5,5), 0)
    flag = 0

    while True:
        _, frame = cap.read()

        flipped = cv2.flip(frame, 1)
        flipped_gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
        #print(frame.shape)   
        flipped_gray = cv2.GaussianBlur(flipped_gray, (5,5), 0)
        difference = cv2.absdiff(first_gray, flipped_gray)
        
        flipped = cv2.circle(flipped, (319,115), 29, (0,255,0), thickness=2)    #upper
        flipped = cv2.circle(flipped, (319,176), 29, (0,255,0), thickness=2)    #lower
        flipped = cv2.circle(flipped, (200,240), 29, (0,255,0), thickness=2)    #left
        flipped = cv2.circle(flipped, (440,240), 29, (0,255,0), thickness=2)    #right
        
        _, difference = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
        difference = cv2.rectangle(difference, (441,211), (499,269), (0,255,0), thickness=2)
        #print(np.sum(difference[211:269, 441:499]))
        if flag == 0:
            sleep(5.0)

        if np.sum(difference[211:269, 411:469]) > 74205:
            #print("right")
            pag.typewrite(['right'])
            sleep(1.0)

        elif np.sum(difference[211:269, 171:229]) > 74205:
            #print("left")
            pag.typewrite(['left'])
            sleep(1.0)
        
        #if np.sum(difference[147:205, 290:348]) < 500:
        elif np.sum(difference[147:205, :]) < 500:
            #print("duck")
            pag.typewrite(['down'])
            sleep(1.0)

        elif np.sum(difference[86:144, 290:348]) > 74205:
            #print("jump")
            pag.typewrite(['up'])
            sleep(1.0)

        else:
            print("-------------")

        cv2.imshow('Live', flipped)
        flag = 1
        #cv2.imshow('first frame', first_frame)
        #cv2.imshow('difference', difference)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()