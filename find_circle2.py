import subprocess
import time
import cv2
import numpy as np
from PIL import Image
import keyboard 

def capture_screenshot(device_id=None):
    adb_command = ["adb"]
    if device_id:
        adb_command += ["-s", device_id]
    adb_command += ["shell", "screencap", "-p"]

    result = subprocess.run(adb_command, stdout=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception("Failed to execute adb command")

    screenshot_data = result.stdout.replace(b'\r\n', b'\n')
    if not screenshot_data:
        raise Exception("Screenshot data is empty")

    image = cv2.imdecode(np.frombuffer(screenshot_data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise Exception("Failed to decode image from screenshot data")

    return image

def extract_and_save_subimages(image):
    # 이미지를 위쪽과 아래쪽 부분으로 분할
    upper_image = image[300:1400, :]
    lower_image = image[1400:2230, :]
    return upper_image, lower_image

def find_pattern(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("유효하지 않은 이미지 형식입니다. NumPy 배열이어야 합니다.")

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # 원 검출
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=20, maxRadius=50)

    # 원이 검출되었는지 확인
    Re_match = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # 원 그리기 (디버그용)
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            
            # 원 중심의 색상 추출
            center_color_bgr = image[y, x]
            center_color_rgb = center_color_bgr[::-1]  # Convert BGR to RGB format
            Re_match.append(((x, y), center_color_rgb.tolist()))
            print(f"원 위치: ({x}, {y}), 색상: {center_color_rgb}")

        groups = group_elements(Re_match)
        sorted_result = sort_groups(groups)

        # for pos, color in sorted_result:
        #     print(f"원 위치: {pos}, 색상: {color}")
        # 원을 그린 이미지를 저장 (디버그용)
        # detected_image_path = f"./detected_circles.png"
        # cv2.imwrite(detected_image_path, image)
        # output_image = Image.open(f"./detected_circles.png")
        # output_image.show()
    else:
        print("원형을 찾을 수 없습니다.")
        return None
    
    return sorted_result

def group_elements(elements):
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

def sort_groups(groups):
    groups.sort(key=lambda g: sum(item[0][1] for item in g) / len(g))
    sorted_result = []
    for group in groups:
        sorted_result.extend(sorted(group, key=lambda x: x[0][0]))
    return sorted_result

def adjust_coordinates(touch_points, section_start_y):
    adjusted_points = [((x, y + section_start_y), color) for (x, y), color in touch_points]
    return adjusted_points

def match_coordinates(upper_touch_points, lower_touch_points):
    new_touch_range = []
    lower_touch_dict = {tuple(color): (x, y) for (x, y), color in lower_touch_points}

    for (x, y), color in upper_touch_points:
        matched_coord = lower_touch_dict.get(tuple(color))
        if matched_coord:
            new_touch_range.append(matched_coord)
        else:
            new_touch_range.append('Nan')
    return new_touch_range

def remove_consecutive_nans(touch_range):
    cleaned_touch_range = []
    previous_value = None
    
    for value in touch_range:
        if value == 'Nan':
            if previous_value != 'Nan':
                cleaned_touch_range.append(value)
        else:
            cleaned_touch_range.append(value)
        previous_value = value

    return cleaned_touch_range

def watch_and_input(touch_range):
    while True:

        if keyboard.is_pressed('left ctrl'):
            if 'Nan' in touch_range:
                nan_index = touch_range.index('Nan')
                coords_to_tap = touch_range[:nan_index]
                for coord in coords_to_tap:
                    if coord != 'Nan':
                        x, y = coord
                        adb_tap(x, y)
                
                # 첫 번째 'Nan' 이전의 좌표를 모두 삭제
                touch_range = touch_range[nan_index+1:]
                
                # 모두 'Nan'이면 스크린샷을 다시 캡처하기 위해 True 반환
                if all(item == 'Nan' for item in touch_range):
                    return True

                # 'Nan'이 없으면 대기
            else:
                for coord in touch_range:
                    if coord != 'Nan':
                        x, y = coord
                        adb_tap(x, y)
                return True  # 모두 'Nan'이 아니면 작업 완료 후 True 반환

    if all(item == 'Nan' for item in touch_range):
        return True  # 모두 'Nan'이면 스크린샷을 다시 캡처하기 위해 True 반환
    else:
        return False

def adb_tap(x, y):
    adb_command = ["adb", "shell", "input", "tap", str(x), str(y)]
    subprocess.run(adb_command)

def remove_consecutive_nans(touch_range):
    cleaned_touch_range = []
    previous_value = None
    
    for value in touch_range:
        if value == 'Nan':
            if previous_value != 'Nan':
                cleaned_touch_range.append(value)
        else:
            cleaned_touch_range.append(value)
        previous_value = value

    return cleaned_touch_range

def main():
    while True:
        image = capture_screenshot(device_id=None)
        upper_image, lower_image = extract_and_save_subimages(image)
        
        upper_touch_points = find_pattern(upper_image)
        if upper_touch_points is None:
            continue
        lower_touch_points = find_pattern(lower_image)
        if lower_touch_points is None:
            continue

        upper_adjusted = adjust_coordinates(upper_touch_points, 300)
        lower_adjusted = adjust_coordinates(lower_touch_points, 1400)
        # Match coordinates
        new_touch_range = match_coordinates(upper_adjusted, lower_adjusted)
        cleaned_touch_range = remove_consecutive_nans(new_touch_range)

        print(cleaned_touch_range)
        if watch_and_input(cleaned_touch_range):
            continue  # 모두 'Nan'이어서 스크린샷을 다시 캡처
        

if __name__ == "__main__":
    main()
