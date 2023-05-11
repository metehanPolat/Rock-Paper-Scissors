import sys
import keras
from random import choice, shuffle, randint  # modele verilen düz makas işaretlerini arttır....

import sys, os, distutils.core

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer

# Configuration setup for testing
cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("kb_tst", )


from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QPushButton,
    QWidget,
    QLabel,

)

from PyQt5.QtGui import QPixmap, QImage, QFont

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import cv2
import numpy as np

DURATION_INT = 8

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors"
}



class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            roi = cv_img # !!!!!!!!!!!!!!!!!!! sanırım burası cv_img[70:300, 10:340]
            if ret:
                self.change_pixmap_signal.emit(roi)


class PredictionThread():
    def __init__(self, cam_image):

        # ilk modeli de burda çalıştırsam ne olur
        predictor = DefaultPredictor(cfg)
        outputs = predictor(cam_image)

        hand_index = outputs["instances"].scores.tolist().index(max(outputs["instances"].scores.tolist()))
        ymin = xmin = ymax = xmax = 0
        output_pred_boxes = outputs["instances"].pred_boxes[hand_index]
        for i in output_pred_boxes.__iter__():
            print(i.cpu().tolist())
            box_list = i.cpu().tolist()
            horizontal_width = round(box_list[2]) - round(box_list[0])
            vertical_width = round(box_list[3]) - round(box_list[1])
            xmin = round((((box_list[2] - box_list[0]) / 2) + box_list[0]) - (((horizontal_width / 2) *0.5 )+(horizontal_width / 2)))
            xmax = round((((box_list[2] - box_list[0]) / 2) + box_list[0]) + (((horizontal_width / 2) *0.5 )+(horizontal_width / 2)))
            ymin = round((((box_list[3] - box_list[1]) / 2) + box_list[1]) - (((vertical_width / 2) *0.5 )+ (vertical_width / 2)))
            ymax = round((((box_list[3] - box_list[1]) / 2) + box_list[1]) + (((vertical_width / 2) *0.5 )+(vertical_width / 2)))
        cropped_image = cam_image[ymin:ymax, xmin:xmax] # ymin, ymax, xmin, xmax

        # --------------------------------------------
        model = keras.models.load_model('game_model99.h5')
        cam_image = cv2.resize(cropped_image, (227, 227))

        pred = model.predict(np.array([cam_image]))
        move_code = np.argmax(pred[0])
        print(move_code)
        self.user_move_name = self.mapper(move_code)

        print(self.user_move_name)

    def get_data(self):
        return self.user_move_name  # initin içerisinden geri dönüş verilemiyor..

    def mapper(self, val):
        return REV_CLASS_MAP[val]


class Window(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()  # kalıtımla aldığı classın initini çalıştırıyor
        QtWidgets.QWidget.__init__(self)
        self.setStyleSheet("background-color: #2B2B2B; color: #D98880;")
        layout = QtWidgets.QGridLayout(self)

        self.playerScore = 0
        self.pcScore = 0

        l = ['rock', 'paper', 'scissors'] * (randint(4, 12))
        shuffle(l)  # liste elemanlarını karıştırır.
        shuffle(l)
        self.computer_move_name = choice(l)

        print(self.computer_move_name)

        # -- Video Kamera --
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)

        # -- Metinler --
        self.layout2 = QtWidgets.QGridLayout()

        self.cam_init = False
        self.cam_image = None

        self.lblPlayer = QLabel()
        self.lblPlayer.setText("Player Move: ")
        self.lblPlayer.setAlignment(Qt.AlignCenter)
        self.lblPlayer.setFont(QFont('Ravie', 20))

        self.lblPC = QLabel()
        self.lblPC.setText("Computer Move: ")
        self.lblPC.setAlignment(Qt.AlignCenter)
        self.lblPC.setFont(QFont('Ravie', 20))

        self.lblWin = QLabel()
        self.lblWin.setText("--  :  --")
        self.lblWin.setAlignment(Qt.AlignCenter)
        self.lblWin.setFont(QFont('Ravie', 20))

        self.lblTime = QLabel()
        # self.lblTime.setText("5")
        self.lblTime.setAlignment(Qt.AlignCenter)
        self.lblTime.setFont(QFont('Ravie', 20))

        # -- Button -- 
        self.btnContinueGame = QtWidgets.QPushButton(self)
        self.btnContinueGame.setText("Next")
        self.btnContinueGame.clicked.connect(self.next_game)
        self.btnContinueGame.hide()

        '''
        self.btnAgainGame = QtWidgets.QPushButton(self)
        self.btnAgainGame.setText("Again")
        self.btnAgainGame.clicked.connect()
        self.btnAgainGame.hide()
        '''

        self.btnStart = QtWidgets.QPushButton(self)
        self.btnStart.setText("Start")
        self.btnStart.clicked.connect(self.start_button_clicked)

        self.btnStart.setStyleSheet("border :5px solid ;"
                             "border-radius: 15px;"
                             "padding: 4px;"
                                    "border-width: 2px;"
                                    "border-style: outset;"
                                    "background-color: gray;"
                                    "weight: 5 px;"
                                    "color: white;"
                                    )
        self.btnStart.setFont(QFont('Ravie', 15))

        self.btnContinueGame.setStyleSheet("border :5px solid ;"
                                    "border-radius: 15px;"
                                    "padding: 4px;"
                                    "border-width: 2px;"
                                    "border-style: outset;"
                                    "background-color: gray;"
                                    "weight: 5 px;"
                                    "color: white;"
                                    )
        self.btnContinueGame.setFont(QFont('Ravie', 15))

        # -- Ara katlar---
        self.layout2.addWidget(self.lblPlayer, 0, 0,1,5)
        self.layout2.addWidget(self.btnStart, 0, 6)
        self.layout2.addWidget(self.btnContinueGame, 0, 6)
        self.btnContinueGame.hide()
        self.layout2.addWidget(self.lblPC, 0, 7,1,5)

        layout3 = QtWidgets.QGridLayout()

        p = QPixmap("paper.jpg")
        p = p.scaledToWidth(440)
        p = p.scaledToHeight(440)

        self.b2 = QLabel()
        self.b2.setAlignment(Qt.AlignCenter)
        self.b2.setPixmap(p)
        self.b2.setFixedHeight(500)
        # self.b2.resize(200, 20)
        layout3.addWidget(self.image_label, 0, 0)
        layout3.addWidget(self.b2, 0, 1)

        layout4 = QtWidgets.QGridLayout()

        layout4.addWidget(self.lblWin, 0, 0)

        # --- Ana layout

        self.time_left_int = DURATION_INT  # kaç saniye olucağı

        layout.addLayout(self.layout2, 0, 0)
        layout.addLayout(layout3, 1, 0)
        layout.addLayout(layout4, 2, 0)
        layout.addLayout(layout4, 3, 0)
        # Set the layout on the application's window
        self.setLayout(layout)
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        # -- Zamanlama fonksiyonları --

    def timer_start(self):
        self.time_left_int = DURATION_INT

        self.my_qtimer = QtCore.QTimer(self)
        self.my_qtimer.timeout.connect(self.update_gui)
        self.my_qtimer.start(1000)

    def update_gui(self):  # textboxu bu fonksiyon değiştiriyor...

        if (self.time_left_int == 0):
            # saniyeyi kapat butonu göster

            self.lblTime.hide()
            self.btnContinueGame.show()

            self.user_move_name = PredictionThread(self.cam_image).get_data()
            self.lblPlayer.setText(f"Player Move : {self.user_move_name}")
            self.lblPC.setText(f"Computer Move : {self.computer_move_name}")
            # print(self.cam_image)# 5 in katı olup oladığına burda bakabilirim...
            self.my_qtimer.stop()
            self.pc_move_image()

        elif (self.time_left_int > 0):

            self.btnContinueGame.hide()

            self.lblTime.show()

            self.time_left_int -= 1

        self.lblTime.setText(str(self.time_left_int))
        # ----

    @pyqtSlot(np.ndarray)  # sinyal gönderme
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        cv_img = cv2.flip(cv_img, 1)
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.cam_image = cv_img  # burdan makine öğrenmesine gönderebilirim
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def start_button_clicked(self):
        # başlatbutonuna basılınca olucaklar..
        self.btnStart.hide()
        # self.btnStart.show()
        self.layout2.addWidget(self.lblTime, 0, 6)
        self.timer_start()
        self.update_gui()

    def score(self):
        if (self.computer_move_name == "rock" and self.user_move_name == "paper"):
            self.playerScore += 1
        elif (self.computer_move_name == "rock" and self.user_move_name == "scissors"):
            self.pcScore += 1
        elif (self.computer_move_name == "paper" and self.user_move_name == "scissors"):
            self.playerScore += 1
        elif(self.computer_move_name == "scissors" and self.user_move_name == "rock"):
            self.playerScore += 1
        elif(self.computer_move_name == "scissors" and self.user_move_name == "paper"):
            self.pcScore += 1
        elif(self.computer_move_name == "paper" and self.user_move_name == "rock"):
            self.pcScore += 1

        self.lblWin.setText(f"{self.playerScore}  :  {self.pcScore}")

    def pc_move_image(self):

        if self.computer_move_name == "rock":
            p = QPixmap("rock.jpg")
        elif self.computer_move_name == "paper":
            p = QPixmap("paper.jpg")
        elif self.computer_move_name == "scissors":
            p = QPixmap("scissors.jpg")
        '''match self.computer_move_name:
            case "rock":
                p = QPixmap("rock.jpg")
            case "paper":
                p = QPixmap("paper.jpg")
            case "scissors":
                p = QPixmap("scissors.jpg")'''

        p = p.scaledToWidth(440)
        p = p.scaledToHeight(440)
        self.b2.setPixmap(p)
        self.score()

    def next_game(self):

        l = ['rock', 'paper', 'scissors'] * (randint(4, 12))
        shuffle(l)  # liste elemanlarını karıştırır.
        shuffle(l)
        self.computer_move_name = choice(l)

        self.time_left_int = DURATION_INT
        self.timer_start()
        self.update_gui()

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setGeometry(300, 150, 1500, 700)
    window.setFixedSize(1500, 700)
    window.show()
    sys.exit(app.exec_())