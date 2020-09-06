import sys
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
import cv2
import eyenput_process
import dlib
import numpy as np

class mainclass(QMainWindow):
    def __init__(self):
        super(mainclass, self).__init__()
        loadUi(r"C:\Users\soura\Desktop\project eyenput\TheEyenput.ui", self)
        self.setWindowIcon(QtGui.QIcon(r"C:\Users\soura\Desktop\project eyenput\logo.png"))
        self.pb_start.clicked.connect(self.startVideo)
        self.pb_stop.clicked.connect(self.stopVideo)
        self.pb_track_left_eye.clicked.connect(self.blob_detection_left)
        self.pb_track_right_eye.clicked.connect(self.blob_detection_right)
        self.camera_is_running = False
        self.fill_count.setAlignment(QtCore.Qt.AlignCenter)

        self.detector_params = cv2.SimpleBlobDetector_Params()
        self.detector_params.filterByArea = True
        self.detector_params.maxArea = 1500
        self.blob_detector = cv2.SimpleBlobDetector_create(self.detector_params)
        
        self.p = r"C:\Users\soura\Desktop\project eyenput\shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.p)
        self.blink = 0
        self.counter = 0
        self.left = 0
        self.right = 0

    @pyqtSlot()
    def startVideo(self):
        if not self.camera_is_running:
            self.cap = cv2.VideoCapture(0)
            self.camera_is_running = True
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.main_stream)
            self.timer.start(1)

    def stopVideo(self):
        if self.camera_is_running:
            self.cap.release()
            self.timer.stop()
            self.camera_is_running = not self.camera_is_running

    def main_stream(self):
        _, self.frame = self.cap.read()
        self.displayImage(self.frame, 0, 1)

        left_eye_threshold = self.thresh_slider_left.value()
        right_eye_threshold = self.thresh_slider_right.value()
        
        face, gray_frame, landmarks = eyenput_process.get_landmarks(self.frame, self.detector, self.predictor)
        self.blink, self.counter = eyenput_process.blink_counter(face, self.blink, self.counter)
        self.fill_count.setText('{}'.format(self.blink))
        left_eye_mod, right_eye_mod = eyenput_process.eyes_preproc(landmarks, self.frame, gray_frame, left_eye_threshold, right_eye_threshold)

        if self.left == 1:
            keypoints = eyenput_process.detect_blob(left_eye_mod, self.blob_detector)
            cv2.drawKeypoints(left_eye_mod, keypoints, left_eye_mod, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            left_eye_mod = np.require(left_eye_mod, np.uint8, 'C')

        self.displayImage(left_eye_mod, 1, 1)
        self.displayImage(right_eye_mod, 2, 1)

    def blob_detection_left(self):
        self.left = 1

    def blob_detection_right(self):
        self.right = 1

    def displayImage(self, frame, flag, window=1):
        qformat = QImage.Format_Indexed8
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        frame = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        frame = frame.rgbSwapped()
        if flag == 0:
            self.frame_window.setPixmap(QPixmap.fromImage(frame))
            self.frame_window.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if flag == 1:
            self.left_eye_window.setPixmap(QPixmap.fromImage(frame))
            self.left_eye_window.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            #self.left_eye_window.setScaledContents(True)
        if flag == 2:
            self.right_eye_window.setPixmap(QPixmap.fromImage(frame))
            self.right_eye_window.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            #self.right_eye_window.setScaledContents(True)


app = QApplication(sys.argv)
window = mainclass()
window.show()
try:
    sys.exit(app.exec_())
except:
    print('Exiting')
