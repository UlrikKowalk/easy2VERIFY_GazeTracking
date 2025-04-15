import time
import yaml
import torch

from easyCNN_01 import easyCNN_01
from Core.VideoSourceMulti import VideoSourceMulti
from Core.FaceMapping import FaceMapping
import cv2
import sys
from PySide6 import QtCore
from PySide6 import QtWidgets
from PySide6 import QtGui
import numpy as np
import os
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

with open('config_testing.yml') as config:
    configuration = yaml.safe_load(config)
    simulation_parameters = configuration['simulation_parameters']

class VideoInputThread(QtCore.QThread):
    new_frame_obtained = QtCore.Signal(np.ndarray, int, float)
    no_camera_signal = QtCore.Signal()
    invalid_file_signal = QtCore.Signal()
    fps_detected = QtCore.Signal(int)
    video_file_ended = QtCore.Signal()

    def __init__(self, cam_id):
        super().__init__()

        self.cam_id = cam_id
        self.input_device = 'camera'
        self.filename = None
        self.fps = None
        self.grayscale = False
        self.is_running = True

    def get_fps(self):
        return self.fps

    def stop(self):
        self.is_running = False

    def set_grayscale(self, grayscale=False):
        self.grayscale = grayscale

    def run(self):

        self.is_running = True
        video_source = VideoSourceMulti(self.cam_id)

        if self.input_device == 'file':
            try:
                self.fps = video_source.start_file(filename=self.filename)
            except:
                self.invalid_file_signal.emit()
        else:
            try:
                self.fps = video_source.start_camera()
            except:
                # no camera found
                self.no_camera_signal.emit()

        if self.fps is not None:
            # use fps if plausible, else assume 30
            self.fps = self.fps if self.fps != 0 else 30

            # send fps value to main app
            self.fps_detected.emit(self.fps)

            start_time = time.time()

            while self.is_running:
                ret, frame = video_source.get_frame()

                if self.grayscale:
                    # for grayscale image
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if ret:
                    self.new_frame_obtained.emit(frame, self.cam_id, self.fps)
                else:
                    self.is_running = False
                    self.video_file_ended.emit()
                time.sleep((1.0 / self.fps) - ((time.time() - start_time) % (1.0 / self.fps)))
            print('Video thread stopped.')

class ArduinoThread(QtCore.QThread):
    arduino_connect = QtCore.Signal()
    arduino_disconnect = QtCore.Signal()

    def __init__(self, num_leds, port='COM3'):
        super().__init__()
        self.wait_interval = 0.02
        self.port = port
        self.arduino = serial.Serial(port=self.port, baudrate=9600, timeout=.1)
        self.is_running = True
        self.num_leds = num_leds
        self.led_pos = int((self.num_leds+1)/2)
        self.is_ready = False
        time.sleep(2)

    def run(self):
        print('Starting Arduino thread.')

        while self.is_running:

            if self.is_ready and not os.path.exists(self.port):
                self.arduino_disconnect.emit()
                self.is_ready = False
            elif not self.is_ready and os.path.exists(self.port):
                if self.arduino.readline().decode().strip() == "READY":
                    self.arduino_connect.emit()
                    self.is_ready = True

            self.arduino.write(f'{self.led_pos}.\n'.encode('utf-8'))
            time.sleep(self.wait_interval)


        print('Arduino thread stopped.')

    def set_data(self, led_pos):
        self.led_pos = int(led_pos)
        # print(f'received new led: {self.led_pos}')

    def stop(self):
        self.is_running = False
        self.arduino.write(f'{-255}.\n'.encode('utf-8'))
        time.sleep(1)
        self.arduino.close()

    def clear_leds(self):
        self.led_pos = -255

    def get_ready(self):
        return self.is_ready

class GazeLive(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GazeLive")
        self.setStyleSheet("background-color: white;")
        self.width_video_full = 320
        self.height_video_full = 240

        self.face_x = 0
        self.face_y = 0
        self.iris_l = [0, 0]
        self.iris_r = [0, 0]
        self.rec_factor = 0.95  # used to smoothen data

        self.is_arduino_ready = False
        self.num_leds = 237
        tmp_scaling_factor = 60 / self.num_leds
        self.min_angle = -45
        self.max_angle = 45

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = "cpu"

        trained_net = f'{simulation_parameters["model"]}'
        print(f"Using device '{self.device}'.")

        # draw mask to focus on eye -> dimensions are known
        width = 60
        height = 60
        center = [30, 30]
        radius = 30
        self.mask = torch.zeros(size=(height, width), dtype=torch.float32, device=self.device)
        for row in range(height):
            for col in range(width):
                if np.sqrt((center[0]-row)**2 + (center[1]-col)**2) <= radius:
                    self.mask[row, col] = 1.0
        # self.mask = np.tile(A=self.mask, reps=(1, 2))

        # load model and push it to device
        map_location = torch.device(self.device)
        sd = torch.load(trained_net, map_location=map_location, weights_only=True)
        self.dnn = easyCNN_01()
        self.dnn.load_state_dict(sd)
        self.dnn.to(self.device)

        self.face_mapping = FaceMapping()
        self.head_azimuth_cam = 0
        self.head_azimuth_cam_absolute = 0
        self.head_elevation_cam = 0
        self.head_roll_cam = 0
        self.face_distance = 1.0

        # create Aruidno thread for led display
        try:
            self.arduino_thread = ArduinoThread(self.num_leds, port="COM5")
            self.arduino_thread.arduino_connect.connect(self.arduino_connect)
            self.arduino_thread.arduino_disconnect.connect(self.arduino_disconnect)
            self.arduino_thread.start()
        except:
            print('No Arduino device found.')

        self.cam_in_use = 0
        # create a separate video thread for each camera
        self.video_thread = VideoInputThread(0)
        self.video_thread.new_frame_obtained.connect(self.new_frame_available)
        self.video_thread.start()
        self.video_thread.set_grayscale(True)

        self.is_first = True
        self.is_cam_running = False
        self.is_error = False
        self.is_blink = False
        self.is_recording = True

        self.kalman = KalmanFilter(dim_x=2, dim_z=1)
        self.kalman.F = np.array([[1., 1.], [0., 1.]])
        self.kalman.H = np.array([[1., 0.]])
        self.kalman.P *= 10
        self.kalman.R = 50
        self.kalman.Q = Q_discrete_white_noise(dim=2, dt=1 / 30, var=0.5)
        self.kalman.x = np.array([0, 0.])

    def __del__(self):
        print('Stopping threads.')
        self.arduino_thread.stop()
        self.video_thread.stop()

    def closeEvent(self, event):
        self.__del__()

    @QtCore.Slot()
    def arduino_connect(self):
        print('Arduino connected.')
        self.is_arduino_ready = True

    @QtCore.Slot()
    def arduino_disconnect(self):
        print('Arduino was disconnected.')
        self.is_arduino_ready = False

    @QtCore.Slot(np.ndarray, int, float)
    def new_frame_available(self, cv_img, cam_id, fps):

        if self.is_cam_running == False:
            self.is_cam_running = True
            # self.check_all_systems()

        # if frame originates from currently used camera, perform face calculations
        if self.cam_in_use == cam_id:
            # if self.is_first:
                # self.start_time = time.time()
            # duration = time.time() - self.start_time
            # self.start_time = time.time()

            if self.is_first:
                self.face_mapping.first(cv_img)
            try:
                self.face_mapping.calculate_face_orientation(cv_img)
                self.head_azimuth_cam, self.head_elevation_cam, self.head_roll_cam, self.face_distance = self.face_mapping.get_position_data()
                _, _, self.is_blink = self.face_mapping.get_gaze()
                self.update_face_coordinates()
                self.is_error = False
            except:
                self.is_error = True

            # convert full image to qt image
            qt_img_face = self.convert_cv_qt(cv_img, cv_img.shape[1], cv_img.shape[0])
            qt_img_eye_left, rect_left = self.crop_eye(qt_img_face, eye='left')
            qt_img_eye_right, rect_right = self.crop_eye(qt_img_face, eye='right')

            # save images of left and right eye to file in user data directory
            if self.is_recording and not self.is_blink:# and self.arduino_thread.get_ready():
                # top, left, height, width
                left_top, left_left, left_height, left_width = rect_left.getRect()
                right_top, right_left, right_height, right_width = rect_right.getRect()

                # filename = self.save_image(cv_img[left_left:left_left + left_height, left_top:left_top + left_width, :],
                #                            cv_img[right_left:right_left + right_height, right_top:right_top + right_width, :])

                # self.list_filename.append(filename)
                # self.list_head_rotation.append(self.head_azimuth_cam)
                # self.list_head_elevation.append(self.head_elevation_cam)
                # self.list_head_roll.append(self.head_roll_cam)

                # led_pos, target = self.get_led_pos_and_target(fps)
                # self.arduino_thread.set_data(led_pos=led_pos)
                # self.list_target.append(target)

                metadata = torch.tensor([self.head_azimuth_cam/180, self.head_elevation_cam/180, self.head_roll_cam/180, self.face_distance/300],
                                        dtype=torch.float32, device=self.device)
                # img_concat = cv2.hconcat([cv_img[left_left:left_left + left_height, left_top:left_top + left_width, :],
                #                           cv_img[right_left:right_left + right_height, right_top:right_top + right_width, :]])

                im_left = cv_img[left_left:left_left + left_height, left_top:left_top + left_width, :]
                im_right = cv_img[right_left:right_left + right_height, right_top:right_top + right_width, :]

                # factor by which to rescale
                scaling_factor = 100.0 / im_left.shape[1]
                img_left_scaled = cv2.resize(im_left, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                img_right_scaled = cv2.resize(im_right, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                im_left = torch.tensor(img_left_scaled, device=self.device, dtype=torch.float32)
                im_right = torch.tensor(img_right_scaled, device=self.device, dtype=torch.float32)

                im_left = im_left[20:-20, 20:-20, 0]# * self.mask
                im_right = im_right[20:-20, 20:-20, 0]# * self.mask
                # execute mask
                # img_scaled *= self.mask
                # plt.imshow(img_scaled)
                # plt.show()
                # image_left = img_scaled[:, :100]
                # image_right = img_scaled[:, 100:]

                im_left = torch.unsqueeze(im_left, dim=0)
                im_left = torch.unsqueeze(im_left, dim=0)
                im_right = torch.unsqueeze(im_right, dim=0)
                im_right = torch.unsqueeze(im_right, dim=0)
                metadata = torch.unsqueeze(metadata, dim=0)

                # result should (!) be -45°...+45° in radian
                result = self.dnn.forward(im_left, im_right, metadata)
                result *= torch.pi
                result *= 2.0

                self.kalman.predict()
                self.kalman.update(result)
                result = self.kalman.x[0]



                result *= self.num_leds
                result += self.num_leds/2
                print(result)
                led_pos = int(result)
                # led_pos = int((result + 45/180*np.pi) / (np.pi/2) * self.num_leds) #Tanh
                # led_pos = int((result + 1) * self.num_leds) #Sigmoid
                # led_pos = int((metadata[0, 0]*2+45)/90 * self.num_leds)

                # sig = [' '*int((result.cpu().detach().numpy()+0.25)*4*30)+'|'+' '*int((0.5-result.cpu().detach().numpy()+0.25)*4*30)]
                # print(sig)
                # print(result)
                # led_pos = result*
                self.arduino_thread.set_data(led_pos=led_pos)

                # increase frame counter by 1
                # self.frame_idx += 1

            # draw camera frame onto correct video monitor
            # self.update_video_display(qt_img_eye_left, qt_img_eye_right)

            # if self.cam_in_use == cam_id and self.is_first:
            #     self.is_first = False
            #     if self.use_head_tracker:
            #         self.headtracker_thread.reset_values()


    def update_face_coordinates(self):
        face_x, face_y = self.face_mapping.get_center()
        iris_l, iris_r, self.is_blink = self.face_mapping.get_iris()

        # if not self.is_blink:
        self.face_x = self.rec_factor * float(face_x) + (1.0 - self.rec_factor) * self.face_x
        self.face_y = self.rec_factor * float(face_y) + (1.0 - self.rec_factor) * self.face_y
        self.iris_l = (self.rec_factor * iris_l[0] + (1.0 - self.rec_factor) * self.iris_l[0],
                       self.rec_factor * iris_l[1] + (1.0 - self.rec_factor) * self.iris_l[1])
        self.iris_r = (self.rec_factor * iris_r[0] + (1.0 - self.rec_factor) * self.iris_r[0],
                       self.rec_factor * iris_r[1] + (1.0 - self.rec_factor) * self.iris_r[1])

    @staticmethod
    def convert_cv_qt(rgb_image, desired_width, desired_height):
        """Convert from an opencv image to QPixmap"""
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        tmp = convert_to_Qt_format.scaled(desired_width, desired_height, QtCore.Qt.KeepAspectRatio,
                                          mode=QtCore.Qt.SmoothTransformation)
        return QtGui.QPixmap.fromImage(tmp)

    def crop_eye(self, frame, eye='left'):

        if eye == 'right':
            center = self.iris_r
        else:
            center = self.iris_l

        # calculate eye width as the distance between both irises
        eye_width = int((self.iris_l[0] - self.iris_r[0]) * 0.8)
        # face detail is rectangular, so use same measurement
        eye_height = eye_width
        # leftmost starting point of face rect
        eye_rect_x = int(min(frame.width(), max(0, int(center[0] - eye_width / 2))))
        # topmost starting point of face rect
        eye_rect_y = int(min(frame.height(), max(0, int(center[1] - eye_height / 2))))
        # create rectangle around face
        temp_rect = QtCore.QRect(eye_rect_x, eye_rect_y, eye_width, eye_height)
        # crop new frame from face rect
        return frame.copy(temp_rect).scaled(self.height_video_full, self.height_video_full,
                                            mode=QtCore.Qt.SmoothTransformation), temp_rect

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    a = GazeLive()
    a.show()
    sys.exit(app.exec())
