import sys
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QGroupBox, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QPlainTextEdit
import threading
from connectATX import ConnectATX
from QPlainTextEditLogger import QPlainTextEditLogger
import logging
from logging.handlers import RotatingFileHandler
from CustomFormatter import CustomFormatter
import os

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.atx_thread = None
        self.atx_instance = None
        self.stop_thread = False
        
    def initUI(self):
        rfh = RotatingFileHandler(filename='./Log.log', mode='a', maxBytes=5 * 1024 * 1024, backupCount=2, encoding=None, delay=0)
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %H:%M:%S', handlers=[rfh])
        
        # 윈도우 설정
        self.setWindowTitle('Auto_MF ver0')
        self.setGeometry(300, 300, 600, 200)

        # 폼 레이아웃
        # 기본적인 입력 필드 설정
        self.input_speed_label = QLabel('입력속도 :', self)
        self.input_speed_edit = QLineEdit(self)
        self.input_row_1 = QLabel('첫줄서클좌표 :', self)
        self.input_row_1_edit = QLineEdit(self)
        self.input_row_2 = QLabel('두번째줄좌표 :', self)
        self.input_row_2_edit = QLineEdit(self)
        self.input_wide = QLabel('서클간격 :', self)
        self.input_wide_edit = QLineEdit(self)
        self.input_updown = QLabel('분할범위 :', self)
        self.input_updown_edit = QLineEdit(self)

        form_layout = QFormLayout()
        form_layout.addRow(self.input_speed_label, self.input_speed_edit)
        form_layout.addRow(self.input_row_1, self.input_row_1_edit)
        form_layout.addRow(self.input_row_2, self.input_row_2_edit)
        form_layout.addRow(self.input_wide, self.input_wide_edit)
        form_layout.addRow(self.input_updown, self.input_updown_edit)

        # 좌표 입력을 위한 필드 (빨강, 주황, 노랑)
        form_layout2 = QFormLayout()
        self.input_touch_Red = QLabel('빨 좌표:', self)
        self.input_touch_Red_edit = QLineEdit(self)
        self.input_touch_Orange = QLabel('주 좌표:', self)
        self.input_touch_Orange_edit = QLineEdit(self)
        self.input_touch_Yellow = QLabel('노 좌표:', self)
        self.input_touch_Yellow_edit = QLineEdit(self)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.input_touch_Red)
        hbox1.addWidget(self.input_touch_Red_edit)
        hbox1.addWidget(self.input_touch_Orange)
        hbox1.addWidget(self.input_touch_Orange_edit)
        hbox1.addWidget(self.input_touch_Yellow)
        hbox1.addWidget(self.input_touch_Yellow_edit)

        form_layout2.addRow(hbox1)

        # RGB 값을 입력하기 위한 추가 필드
        self.input_touch_Red_rgb = QLabel('빨 RGB:', self)
        self.input_touch_Red_rgb_edit = QLineEdit(self)
        self.input_touch_Orange_rgb = QLabel('주 RGB:', self)
        self.input_touch_Orange_rgb_edit = QLineEdit(self)
        self.input_touch_Yellow_rgb = QLabel('노 RGB:', self)
        self.input_touch_Yellow_rgb_edit = QLineEdit(self)

        hbox_rgb = QHBoxLayout()
        hbox_rgb.addWidget(self.input_touch_Red_rgb)
        hbox_rgb.addWidget(self.input_touch_Red_rgb_edit)
        hbox_rgb.addWidget(self.input_touch_Orange_rgb)
        hbox_rgb.addWidget(self.input_touch_Orange_rgb_edit)
        hbox_rgb.addWidget(self.input_touch_Yellow_rgb)
        hbox_rgb.addWidget(self.input_touch_Yellow_rgb_edit)

        form_layout2.addRow(hbox_rgb)

        # form_layout3 설정 (초록, 파랑, 핑크 좌표)
        form_layout3 = QFormLayout()
        self.input_touch_green = QLabel('초 좌표:', self)
        self.input_touch_green_edit = QLineEdit(self)
        self.input_touch_blue = QLabel('파 좌표:', self)
        self.input_touch_blue_edit = QLineEdit(self)
        self.input_touch_pink = QLabel('핑 좌표:', self)
        self.input_touch_pink_edit = QLineEdit(self)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.input_touch_green)
        hbox2.addWidget(self.input_touch_green_edit)
        hbox2.addWidget(self.input_touch_blue)
        hbox2.addWidget(self.input_touch_blue_edit)
        hbox2.addWidget(self.input_touch_pink)
        hbox2.addWidget(self.input_touch_pink_edit)

        form_layout3.addRow(hbox2)

        # RGB 값을 입력하기 위한 추가 필드 (초록, 파랑, 핑크)
        self.input_touch_green_rgb = QLabel('초 RGB:', self)
        self.input_touch_green_rgb_edit = QLineEdit(self)
        self.input_touch_blue_rgb = QLabel('파 RGB:', self)
        self.input_touch_blue_rgb_edit = QLineEdit(self)
        self.input_touch_pink_rgb = QLabel('핑 RGB:', self)
        self.input_touch_pink_rgb_edit = QLineEdit(self)

        hbox_rgb2 = QHBoxLayout()
        hbox_rgb2.addWidget(self.input_touch_green_rgb)
        hbox_rgb2.addWidget(self.input_touch_green_rgb_edit)
        hbox_rgb2.addWidget(self.input_touch_blue_rgb)
        hbox_rgb2.addWidget(self.input_touch_blue_rgb_edit)
        hbox_rgb2.addWidget(self.input_touch_pink_rgb)
        hbox_rgb2.addWidget(self.input_touch_pink_rgb_edit)

        form_layout3.addRow(hbox_rgb2)

        self.input_pixel_Red = QLabel('빨 픽셀 RGB:', self)
        self.input_pixel_Red_edit = QLineEdit(self)
        self.input_pixel_Orange = QLabel('주 픽셀 RGB:', self)
        self.input_pixel_Orange_edit = QLineEdit(self)
        self.input_pixel_Yellow = QLabel('노 픽셀 RGB:', self)
        self.input_pixel_Yellow_edit = QLineEdit(self)
        self.input_pixel_Purple = QLabel('보 픽셀 RGB:', self)
        self.input_pixel_Purple_edit = QLineEdit(self)
        self.input_pixel_Green = QLabel('초 픽셀 RGB:', self)
        self.input_pixel_Green_edit = QLineEdit(self)
        self.input_pixel_Blue = QLabel('파 픽셀 RGB:', self)
        self.input_pixel_Blue_edit = QLineEdit(self)
        self.input_pixel_Pink = QLabel('핑 픽셀 RGB:', self)
        self.input_pixel_Pink_edit = QLineEdit(self)
        self.input_pixel_Lime = QLabel('녹 픽셀 RGB:', self)
        self.input_pixel_Lime_edit = QLineEdit(self)

        # 픽셀 RGB 값을 위한 레이아웃 추가
        hbox_pixel_rgb = QHBoxLayout()
        hbox_pixel_rgb.addWidget(self.input_pixel_Red)
        hbox_pixel_rgb.addWidget(self.input_pixel_Red_edit)
        hbox_pixel_rgb.addWidget(self.input_pixel_Orange)
        hbox_pixel_rgb.addWidget(self.input_pixel_Orange_edit)
        hbox_pixel_rgb.addWidget(self.input_pixel_Yellow)
        hbox_pixel_rgb.addWidget(self.input_pixel_Yellow_edit)

        hbox_pixel_rgb2 = QHBoxLayout()
        hbox_pixel_rgb2.addWidget(self.input_pixel_Purple)
        hbox_pixel_rgb2.addWidget(self.input_pixel_Purple_edit)
        hbox_pixel_rgb2.addWidget(self.input_pixel_Green)
        hbox_pixel_rgb2.addWidget(self.input_pixel_Green_edit)
        hbox_pixel_rgb2.addWidget(self.input_pixel_Blue)
        hbox_pixel_rgb2.addWidget(self.input_pixel_Blue_edit)

        hbox_pixel_rgb3 = QHBoxLayout()
        hbox_pixel_rgb3.addWidget(self.input_pixel_Pink)
        hbox_pixel_rgb3.addWidget(self.input_pixel_Pink_edit)
        hbox_pixel_rgb3.addWidget(self.input_pixel_Lime)
        hbox_pixel_rgb3.addWidget(self.input_pixel_Lime_edit)

        # Form Layout에 추가된 픽셀 RGB 값 입력 필드를 추가
        form_layout3.addRow(hbox_pixel_rgb)
        form_layout3.addRow(hbox_pixel_rgb2)
        form_layout3.addRow(hbox_pixel_rgb3)

        self.A_label2 = QLabel('적정입력속도 : 0.05 (단위:s)', self)
        
        # 실행 및 중단 버튼
        self.run_button = QPushButton('실행', self)
        self.stop_button = QPushButton('중단', self)

        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)

        # 메인 레이아웃
        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addLayout(form_layout2)
        main_layout.addLayout(form_layout3)
        main_layout.addWidget(self.A_label2)
        main_layout.addLayout(button_layout)

        self.groupBox2 = QGroupBox("Log")
        self.logBrowser = QPlainTextEditLogger(self.groupBox2)
        self.logBrowser.setFormatter(CustomFormatter())
        logging.getLogger().addHandler(self.logBrowser)
        logging.getLogger().setLevel(logging.INFO)
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.logBrowser.widget)
        self.groupBox2.setLayout(log_layout)

        main_layout.addWidget(self.groupBox2)
        self.load_data()

        self.setLayout(main_layout)
        self.isDebug = True
        self.thread = QThread()
        
        # 버튼 동작 연결
        logging.info('프로그램이 정상 기동했습니다')
        self.run_button.clicked.connect(self.run)
        self.stop_button.clicked.connect(self.stop)

    def enableRunBtn(self):
        self.run_button.setEnabled(True)
        self.run_button.setText('실행')

    # 실행버튼 비활성화
    def disableRunBtn(self):
        self.run_button.setEnabled(False)
        self.run_button.setText('실행 중')

    def stop(self):
        if self.thread is not None:
            self.stop_thread = True
            if self.thread.isRunning():
                self.thread.quit()
                self.thread.wait()
            self.thread = None
            self.atx_instance = None
            logging.info("작업 중단")
            self.enableRunBtn()

    def showinfo(self, str):
        logging.error(f'알림 - {str}')
    
    def ShowError(self, str):
        logging.error(f'프로그램에러 - {str}')
        self.enableRunBtn()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()

    def run(self):
        self.disableRunBtn()
        try:
            logging.info('프로그램 실행')  # Log message
            
            # 입력된 데이터를 가져오기
            input_speed = self.input_speed_edit.text()
            input_row_1 = self.input_row_1_edit.text()
            input_row_2 = self.input_row_2_edit.text()
            input_wide = self.input_wide_edit.text()
            input_updown = self.input_updown_edit.text()
            
            # 색상 좌표 및 RGB 값 가져오기
            input_touch_Red = self.input_touch_Red_edit.text()
            input_touch_Orange = self.input_touch_Orange_edit.text()
            input_touch_Yellow = self.input_touch_Yellow_edit.text()
            input_touch_green = self.input_touch_green_edit.text()
            input_touch_blue = self.input_touch_blue_edit.text()
            input_touch_pink = self.input_touch_pink_edit.text()

            input_touch_Red_rgb = self.input_touch_Red_rgb_edit.text()
            input_touch_Orange_rgb = self.input_touch_Orange_rgb_edit.text()
            input_touch_Yellow_rgb = self.input_touch_Yellow_rgb_edit.text()
            input_touch_green_rgb = self.input_touch_green_rgb_edit.text()
            input_touch_blue_rgb = self.input_touch_blue_rgb_edit.text()
            input_touch_pink_rgb = self.input_touch_pink_rgb_edit.text()

            input_pixel_Red = self.input_pixel_Red_edit.text()
            input_pixel_Orange = self.input_pixel_Orange_edit.text()
            input_pixel_Yellow = self.input_pixel_Yellow_edit.text()
            input_pixel_Purple = self.input_pixel_Purple_edit.text()
            input_pixel_Green = self.input_pixel_Green_edit.text()
            input_pixel_Blue = self.input_pixel_Blue_edit.text()
            input_pixel_Pink = self.input_pixel_Pink_edit.text()
            input_pixel_Lime = self.input_pixel_Lime_edit.text()

            with open('temp.txt', 'w') as file:
                file.write(f"{input_speed}\n")
                file.write(f"{input_row_1}\n")
                file.write(f"{input_row_2}\n")
                file.write(f"{input_wide}\n")
                file.write(f"{input_updown}\n")
                file.write(f"{input_touch_Red}\n")
                file.write(f"{input_touch_Orange}\n")
                file.write(f"{input_touch_Yellow}\n")
                file.write(f"{input_touch_green}\n")
                file.write(f"{input_touch_blue}\n")
                file.write(f"{input_touch_pink}\n")
                file.write(f"{input_touch_Red_rgb}\n")
                file.write(f"{input_touch_Orange_rgb}\n")
                file.write(f"{input_touch_Yellow_rgb}\n")
                file.write(f"{input_touch_green_rgb}\n")
                file.write(f"{input_touch_blue_rgb}\n")
                file.write(f"{input_touch_pink_rgb}\n")
                file.write(f"{input_pixel_Red}\n")
                file.write(f"{input_pixel_Orange}\n")
                file.write(f"{input_pixel_Yellow}\n")
                file.write(f"{input_pixel_Purple}\n")
                file.write(f"{input_pixel_Green}\n")
                file.write(f"{input_pixel_Blue}\n")
                file.write(f"{input_pixel_Pink}\n")
                file.write(f"{input_pixel_Lime}\n")

            if self.atx_thread is None:
                self.stop_thread = False
                
                # ConnectATX 클래스의 인스턴스를 생성할 때 추가된 변수를 전달합니다.
                self.atx_instance = ConnectATX(
                    self.isDebug,
                    input_speed,
                    input_row_1,
                    input_row_2,
                    input_wide,
                    input_updown,
                    input_touch_Red,
                    input_touch_Orange,
                    input_touch_Yellow,
                    input_touch_green,
                    input_touch_blue,
                    input_touch_pink,
                    input_touch_Red_rgb,
                    input_touch_Orange_rgb,
                    input_touch_Yellow_rgb,
                    input_touch_green_rgb,
                    input_touch_blue_rgb,
                    input_touch_pink_rgb,
                    input_pixel_Red,
                    input_pixel_Orange,
                    input_pixel_Yellow,
                    input_pixel_Purple,
                    input_pixel_Green,
                    input_pixel_Blue,
                    input_pixel_Pink,
                    input_pixel_Lime
                )
                self.atx_instance.moveToThread(self.thread)
                self.atx_instance.returnInfo.connect(self.showinfo)
                self.atx_instance.returnError.connect(self.ShowError)
                self.thread.started.connect(self.atx_instance.run)
                self.thread.start()
        except Exception as e:
            logging.info(f'Error: {e}')

    def load_data(self):
        #""" 프로그램 시작 시 temp.txt에서 데이터를 불러와 입력 필드를 채움 """
        if os.path.exists('temp.txt'):
            with open('temp.txt', 'r') as file:
                lines = file.readlines()
                self.input_speed_edit.setText(lines[0].strip())
                self.input_row_1_edit.setText(lines[1].strip())
                self.input_row_2_edit.setText(lines[2].strip())
                self.input_wide_edit.setText(lines[3].strip())
                self.input_updown_edit.setText(lines[4].strip())
                self.input_touch_Red_edit.setText(lines[5].strip())
                self.input_touch_Orange_edit.setText(lines[6].strip())
                self.input_touch_Yellow_edit.setText(lines[7].strip())
                self.input_touch_green_edit.setText(lines[8].strip())
                self.input_touch_blue_edit.setText(lines[9].strip())
                self.input_touch_pink_edit.setText(lines[10].strip())
                self.input_touch_Red_rgb_edit.setText(lines[11].strip())
                self.input_touch_Orange_rgb_edit.setText(lines[12].strip())
                self.input_touch_Yellow_rgb_edit.setText(lines[13].strip())
                self.input_touch_green_rgb_edit.setText(lines[14].strip())
                self.input_touch_blue_rgb_edit.setText(lines[15].strip())
                self.input_touch_pink_rgb_edit.setText(lines[16].strip())
                self.input_pixel_Red_edit.setText(lines[17].strip())
                self.input_pixel_Orange_edit.setText(lines[18].strip())
                self.input_pixel_Yellow_edit.setText(lines[19].strip())
                self.input_pixel_Purple_edit.setText(lines[20].strip())
                self.input_pixel_Green_edit.setText(lines[21].strip())
                self.input_pixel_Blue_edit.setText(lines[22].strip())
                self.input_pixel_Pink_edit.setText(lines[23].strip())
                self.input_pixel_Lime_edit.setText(lines[24].strip())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MyApp()
    ui.show()
    sys.exit(app.exec_())
