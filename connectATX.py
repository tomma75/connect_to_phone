import math
import os
import time
from PyQt5.QtCore import QObject, pyqtSignal
import cv2
import numpy as np
from PIL import Image
import keyboard
import uiautomator2 as u2
import debugpy
import pandas as pd
from sklearn.cluster import KMeans
from numba import jit

class ConnectATX(QObject):
    returnInfo = pyqtSignal(str)
    returnError = pyqtSignal(Exception)

    def __init__(self, isDebug, input_speed, input_row_1, input_row_2, input_wide, input_updown,
             input_touch_Red, input_touch_Orange, input_touch_Yellow,
             input_touch_green, input_touch_blue, input_touch_pink,
             input_touch_Red_rgb, input_touch_Orange_rgb, input_touch_Yellow_rgb,
             input_touch_green_rgb, input_touch_blue_rgb, input_touch_pink_rgb,
             input_pixel_Red, input_pixel_Orange, input_pixel_Yellow,
             input_pixel_Purple, input_pixel_Green, input_pixel_Blue,
             input_pixel_Pink, input_pixel_Lime):
        super().__init__()
        self.isDebug = isDebug
        self.input_speed = input_speed
        self.input_row_1 = input_row_1
        self.input_row_2 = input_row_2
        self.input_wide = input_wide
        self.input_updown = input_updown

        self.input_touch_Red = input_touch_Red
        self.input_touch_Orange = input_touch_Orange
        self.input_touch_Yellow = input_touch_Yellow
        self.input_touch_green = input_touch_green
        self.input_touch_blue = input_touch_blue
        self.input_touch_pink = input_touch_pink

        self.input_touch_Red_rgb = input_touch_Red_rgb
        self.input_touch_Orange_rgb = input_touch_Orange_rgb
        self.input_touch_Yellow_rgb = input_touch_Yellow_rgb
        self.input_touch_green_rgb = input_touch_green_rgb
        self.input_touch_blue_rgb = input_touch_blue_rgb
        self.input_touch_pink_rgb = input_touch_pink_rgb

        # 추가된 픽셀 RGB 값을 클래스 속성으로 저장
        self.input_pixel_Red = input_pixel_Red
        self.input_pixel_Orange = input_pixel_Orange
        self.input_pixel_Yellow = input_pixel_Yellow
        self.input_pixel_Blue = input_pixel_Blue
        self.input_pixel_Green = input_pixel_Green
        self.input_pixel_Purple = input_pixel_Purple
        self.input_pixel_Pink = input_pixel_Pink
        self.input_pixel_Lime = input_pixel_Lime

        # 디바이스 연결 설정
        self.device = self.connect_device()

    def run(self):
        try:
            pd.set_option('mode.chained_assignment', None)
            if self.isDebug:
                debugpy.debug_this_thread()
            device_id = None  # 필요 시 특정 장치 ID로 변경
            d = self.connect_device(device_id)
            color_map = self.initialize_color_map() 
            pattern_path = r'.\RB.jpg'  # RB.jpg 이미지 경로
            pattern_path2 = r'.\Pig.jpg'  # RB.jpg 이미지 경로
            pattern = cv2.imread(pattern_path, cv2.IMREAD_COLOR)
            pattern2 = cv2.imread(pattern_path2, cv2.IMREAD_COLOR)
            while True:
                image = self.capture_screenshot(d)
                up, down = self.input_updown.split(',')
                up = int(up)
                down = int(down)
                upper_image, lower_image = self.extract_and_save_subimages(image,up,down)
                cv2.imwrite("./upper.jpg", upper_image)
                cv2.imwrite("./lower.jpg", lower_image)
                Match = self.find_pattern3(lower_image, pattern,pattern2)
                if Match is None:
                    continue
                upper_touch_points, trigger = self.find_pattern2(upper_image, Match, up,color_map)
                if trigger == False: 
                    continue
                self.find_matching_points(upper_touch_points)
        
        except Exception as e:  
            self.returnError.emit(e)                    

    def log_to_file(self, message, file_path='./log.txt'):
        with open(file_path, 'a') as log_file:  # 'a' 모드는 파일에 내용을 추가
            print(message, file=log_file)

    def connect_device(self, device_id=None):
        if device_id:
            d = u2.connect(device_id)
        else:
            d = u2.connect()
        return d

    def capture_screenshot(self, d):
        screenshot_data = d.screenshot(format='opencv')
        if screenshot_data is None:
            raise Exception("Failed to capture screenshot")
        screenshot_data = self.increase_saturation(screenshot_data, saturation_scale=1.5)
        return screenshot_data

    def extract_and_save_subimages(self,image,up,down):
        # 이미지를 위쪽과 아래쪽 부분으로 분할
        upper_image = image[up:down, :]
        lower_image = image[down:2230, :]
        return upper_image, lower_image

    def initialize_color_map(self):
        # /로 구분된 RGB 값을 분리하여 각 터치와 연결
        color_inputs = {
            'Red': (self.input_pixel_Red, self.input_touch_Red),
            'Orange': (self.input_pixel_Orange, self.input_touch_Orange),
            'Yellow': (self.input_pixel_Yellow, self.input_touch_Yellow),
            'Green': (self.input_pixel_Green, self.input_touch_green),
            'Blue': (self.input_pixel_Blue, self.input_touch_blue),
            'Pink': (self.input_pixel_Pink, self.input_touch_pink),
            'Purple': (self.input_pixel_Purple, self.input_touch_pink),  # 보라색 설정
            'Lime': (self.input_pixel_Lime, self.input_touch_Yellow)  # 녹색 설정
        }

        # 색상별 RGB 값과 터치 매핑을 저장할 딕셔너리
        color_map = {}

        # 각 색상에 대해 반복하여 RGB와 터치 값을 매핑
        for color_name, (rgb_string, touch_value) in color_inputs.items():
            # /로 구분된 RGB 값을 분리
            rgb_values = rgb_string.split('/')

            # 각 RGB 값에 대해 매핑
            for idx, rgb in enumerate(rgb_values):
                # 색상 이름에 인덱스를 붙여서 고유하게 만듭니다.
                unique_color_name = f"{color_name}_{idx}"
                color_map[unique_color_name] = (rgb, touch_value)

        return color_map

    def find_pattern2(self, image, Match, up, color_map):
        # input_row_1과 input_row_2는 'x,y' 형태의 문자열로 가정합니다.
        x_start, y_start = map(int, self.input_row_1.split(','))
        _, y_offset = map(int, self.input_row_2.split(','))
        y_offset = y_offset - y_start
        wide = int(self.input_wide)

        upper_touch_points = ""
        matched = False
        for i in range(3):  # 3개의 줄을 읽기 위해 3번 반복
            row_points = ""  # 현재 줄의 결과를 저장할 문자열

            for j in range(6):  # 각 줄에서 6개 좌표를 읽습니다.
                x1 = x_start + j * wide
                y = y_start + i * y_offset - up  # i에 따라 y좌표를 증가시킴

                # 현재 줄에서 좌표의 RGB 값 읽기
                bgr_value = image[y, x1]
                rgb_value = tuple(bgr_value[::-1])  # BGR -> RGB 변환
                self.log_to_file(f"좌표값 : {x1}, {y} RGB값 : {rgb_value}")

                # RedSet 또는 GreenSet에 해당하는 색상 비교 
                for color_name, (rgb, touch_value) in color_map.items():
                    # RGB 값이 color_map에 있는지 확인
                    expected_rgb = tuple(map(int, rgb.split(',')))
                    if all(abs(rv - ev) <= 8 for rv, ev in zip(rgb_value, expected_rgb)):
                        # RGB 값이 있으면 해당하는 Set에 속하는지 확인
                        if Match == 'RedSet':
                            # RedSet에 포함된 색상('Red', 'Orange', 'Yellow', 'Purple') 확인
                            if any(key in color_name for key in ['Red', 'Orange', 'Yellow', 'Purple']):
                                row_points += f"{touch_value}/"  # 해당 좌표를 추가
                                matched = True
                                break
                            else:
                                row_points += '-/'  # Set에 속하지 않으면 '-'
                        elif Match == 'GreenSet':
                            # GreenSet에 포함된 색상('Green', 'Blue', 'Pink', 'Lime') 확인
                            if any(key in color_name for key in ['Green', 'Blue', 'Pink', 'Lime']):
                                row_points += f"{touch_value}/"  # 해당 좌표를 추가
                                matched = True
                                break
                            else:
                                row_points += '-/'  # Set에 속하지 않으면 '-'

            if i == 1 and matched == False:
                return None, False

            # 현재 줄의 결과를 전체 결과에 추가
            upper_touch_points += row_points

        # 연속된 '-' 값 병합
        merged_points = self.merge_hyphens(upper_touch_points.split('/'))  # 문자열을 '/'로 분할하여 병합

        return merged_points, matched

    def merge_hyphens(self, points_str):
        # 입력 문자열을 '/'로 분할하여 리스트로 변환
        
        # 연속된 '-' 값을 하나로 병합
        merged_points = []
        hyphen_sequence = False

        for point in points_str:
            if point == '-':
                if not hyphen_sequence:
                    merged_points.append('-')
                    hyphen_sequence = True
            else:
                merged_points.append(point)
                hyphen_sequence = False

        # 병합된 리스트를 다시 '/'로 연결된 문자열로 변환하여 반환
        return '/'.join(merged_points)

    def find_pattern3(self, image, pattern, pattern2):
        # 색상별 좌표와 RGB 값 설정
        result = cv2.matchTemplate(image, pattern, cv2.TM_CCOEFF_NORMED)
        result2 = cv2.matchTemplate(image, pattern2, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result2)

        # 매칭 임계값 설정 (임계값은 조정 가능)
        threshold = 0.8

        if max_val >= threshold:
            return 'GreenSet'
        elif max_val2 >= threshold:
            return 'RedSet'
        else:
            return None

    def find_pattern(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("유효하지 않은 이미지 형식입니다. NumPy 배열이어야 합니다.")

        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # 적응형 임계처리 적용
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # 모폴로지 변환 적용 (열림 연산)
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

        # 원 검출
        circles = cv2.HoughCircles(morph, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                param1=50, param2=30, minRadius=30, maxRadius=70)
        # 원이 검출되었는지 확인
        Re_match = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # 원의 중심 주변 픽셀들의 RGB 값을 추출
                mask = np.zeros_like(gray)
                cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
                masked_image = cv2.bitwise_and(image, image, mask=mask)

                # 원의 영역에서 RGB 값을 추출하고 k-means 클러스터링 적용
                pixel_values = masked_image[mask == 255].reshape(-1, 3)
                if len(pixel_values) == 0:
                    continue

                kmeans = KMeans(n_clusters=1, random_state=0).fit(pixel_values)
                dominant_color = kmeans.cluster_centers_[0]
                rounded_color = [float(np.round(c)) for c in dominant_color]

                # RGB 값이 모두 같은 경우 무시
                if rounded_color[0] == rounded_color[1] == rounded_color[2]:
                    continue

                cv2.circle(image, (x, y), r, (0, 255, 0), 4)

                Re_match.append(((x, y), rounded_color))
                print(f"원 위치: ({x}, {y}), 주요 색상: {rounded_color}")

            groups = self.group_elements(Re_match)
            sorted_result = self.sort_groups(groups)

        else:
            print("원형을 찾을 수 없습니다.")
            return None

        return sorted_result

    def increase_saturation(self, image, saturation_scale=8):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        # 채도 값을 scale 배 증가시킴
        s = cv2.multiply(s, saturation_scale)
        s = np.clip(s, 0, 255).astype(hsv_image.dtype)

        # HSV 이미지를 다시 합침
        hsv_image = cv2.merge([h, s, v])
        enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        return enhanced_image
    def group_elements(self, elements):
        if not elements:  # elements 리스트가 비어있는지 확인
            return []

        elements.sort(key=lambda x: x[0][1])
        groups = []
        current_group = [elements[0]]
        for item in elements[1:]:
            if abs(item[0][1] - current_group[-1][0][1]) <= 20:
                current_group.append(item)
            else:
                groups.append(current_group)
                current_group = [item]
        groups.append(current_group)
        return groups

    def sort_groups(self,groups):
        groups.sort(key=lambda g: sum(item[0][1] for item in g) / len(g))
        sorted_result = []
        for group in groups:
            sorted_result.extend(sorted(group, key=lambda x: x[0][0]))
        return sorted_result

    def adjust_coordinates(self,touch_points, section_start_y):
        adjusted_points = [((x, y + section_start_y), color) for (x, y), color in touch_points]
        return adjusted_points

    def match_coordinates(self,upper_touch_points, lower_touch_points):
        new_touch_range = []
        lower_touch_dict = {tuple(color): (x, y) for (x, y), color in lower_touch_points}

        for (x, y), color in upper_touch_points:
            matched_coord = lower_touch_dict.get(tuple(color))
            if matched_coord:
                new_touch_range.append(matched_coord)
            else:
                new_touch_range.append('Nan')
        return new_touch_range

    def remove_consecutive_nans(self,touch_range):
        cleaned_touch_range = []
        previous_nan = False
        
        for value in touch_range:
            if isinstance(value, tuple) and math.isnan(value[0]) and math.isnan(value[1]):
                if not previous_nan:
                    cleaned_touch_range.append(value)
                    previous_nan = True
            else:
                cleaned_touch_range.append(value)
                previous_nan = False

        return cleaned_touch_range

        
    def watch_and_input(self, touch_range, d):
        while touch_range:
            coords_to_click = []

            # Step 1: Read touch_range.
            while touch_range:
                coord = touch_range[0]
                if isinstance(coord, tuple) and not math.isnan(coord[0]) and not math.isnan(coord[1]):
                    coords_to_click.append((float(coord[0]), float(coord[1])))
                    touch_range.pop(0)
                else:
                    break

            # Step 2: If there are coordinates to click, click them in order.
            for x, y in coords_to_click:
                self.adb_tap(x, y)
                time.sleep(self.input_speed)

            # Step 3: Clear the coords_to_click list.
            coords_to_click.clear()
            
            # Step 4: Wait for left ctrl click.
            while not keyboard.is_pressed('left ctrl'):
                time.sleep(0.05)

            # Step 5: If I click left ctrl, click next coordinates until get nan.
            if touch_range:
                coord = touch_range.pop(0)  # Safely pop the first element
                if touch_range:
                    coord = touch_range[0]
                    while isinstance(coord, tuple) and not math.isnan(coord[0]) and not math.isnan(coord[1]):
                        coords_to_click.append((float(coord[0]), float(coord[1])))
                        touch_range.pop(0)
                        if touch_range:
                            coord = touch_range[0]
                        else:
                            break
                    for x, y in coords_to_click:
                        self.adb_tap(x, y)
                        time.sleep(self.input_speed)
                    coords_to_click.clear()
            
        # Step 6: If no data in touch_range, return True.
        return True


    def color_distance(self,color1, color2, max_individual_difference=30, max_total_difference=40):
        individual_differences = [abs(c1 - c2) for c1, c2 in zip(color1, color2)]
        total_difference = sum(individual_differences)
        return all(d <= max_individual_difference for d in individual_differences) and total_difference <= max_total_difference

    def find_matching_points(self, upper_touch_points):
        # 터치 명령어 문자열 초기화
        touch_commands = ""

        # 입력 문자열을 '/'로 분할하여 리스트로 변환
        points = upper_touch_points.split('/')

        for point in points:
            if point == '-':
                # '-'가 나왔을 때 'space bar' 입력 대기
                if touch_commands:  # 대기 전에 저장된 명령어 실행
                    os.system(touch_commands.rstrip(' && '))  # 마지막 '&&' 제거 후 실행
                    touch_commands = ""  # 터치 명령어 문자열 초기화

                print("Waiting for space bar input...")
                keyboard.wait('space')
            elif point.strip():  # 빈 문자열을 제외한 경우만 처리
                try:
                    # 좌표가 있는 경우 adb_tap 명령어 추가
                    x, y = map(int, point.split(','))  # 좌표를 정수로 변환
                    touch_commands += f"adb shell input tap {x} {y} && "
                except ValueError:
                    print(f"Invalid point value: {point}")  # 잘못된 입력값이 있는 경우 알림

        # 마지막에 남아 있는 터치 명령어가 있으면 실행
        if touch_commands:
            os.system(touch_commands.rstrip(' && '))  # 마지막 '&&' 제거 후 실행

