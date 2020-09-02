import sys
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
import cv2
import eyenput_process

class mainclass(QDialog):
    def __init__(self):
        super(mainclass, self).__init__()
        loadUi(r"C:\Users\soura\Desktop\project eyenput\TheEyenput.ui", self)
        self.pb_start.clicked.connect(self.startVideo)
        self.pb_stop.clicked.connect(self.stopVideo)
        self.pb_track_eyes.clicked.connect(self.track_eyes)
        self.camera_is_running = False
        self.fill_count.setAlignment(QtCore.Qt.AlignCenter)
        self.fill_count.setText('0')

    @pyqtSlot()
    def startVideo(self):
        if not self.camera_is_running:
            self.cap = cv2.VideoCapture(0)
            self.camera_is_running = True
            while self.cap.isOpened():
                _, self.frame = self.cap.read()
                self.displayImage(self.frame, 0, 1)
                cv2.waitKey()
            cv2.destroyAllWindows()

    def stopVideo(self):
        if self.camera_is_running:
            self.cap.release()
            window.close()

    def track_eyes(self):
        left_eye_threshold = self.left_eye_thresh_slider.value()
        right_eye_threshold = self.right_eye_thresh_slider.value()
        while self.cap.isOpened():
            face, gray_frame, landmarks = eyenput_process.get_landmarks(self.frame, left_eye_threshold)
            left_eye_mod, right_eye_mod = eyenput_process.eyes_preproc(landmarks, self.frame, gray_frame, left_eye_threshold)
            self.displayImage(left_eye_mod, 1, 1)
            self.displayImage(right_eye_mod, 2, 1)
            cv2.waitKey()

    def displayImage(self, frame, flag, window=1):
        qformat = QImage.Format_Indexed8

        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        frame = QImage(frame, frame.shape[1], frame.shape[0], qformat)
        frame = frame.rgbSwapped()
        if flag == 0:
            self.frame_window.setPixmap(QPixmap.fromImage(frame))
            self.frame_window.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if flag == 1:
            self.left_eye_window.setPixmap(QPixmap.fromImage(frame))
            #self.left_eye_window.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.left_eye_window.setScaledContents(True)
        if flag == 2:
            self.right_eye_window.setPixmap(QPixmap.fromImage(frame))
            #self.right_window.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.right_eye_window.setScaledContents(True)
app = QApplication(sys.argv)
window = mainclass()
window.show()
try:
    sys.exit(app.exec_())
except:
    print('Exiting')
