import math
import os
import time
import cv2
import numpy as np
from PIL import Image
import keyboard
import uiautomator2 as u2

def connect_device(device_id=None):
    if device_id:
        d = u2.connect(device_id)
    else:
        d = u2.connect()
    return d

def capture_screenshot(d):
    screenshot_data = d.screenshot(format='opencv')
    if screenshot_data is None:
        raise Exception("Failed to capture screenshot")
    return screenshot_data

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
            center_color_bgr = image[y, x]
            center_color_rgb = center_color_bgr[::-1]  # Convert BGR to RGB format

            # RGB 값이 모두 같은 경우 무시
            if center_color_rgb[0] == center_color_rgb[1] == center_color_rgb[2]:
                continue

            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            
            # 원 중심의 색상 추출
            center_color_bgr = image[y, x]
            center_color_rgb = center_color_bgr[::-1]  # Convert BGR to RGB format
            Re_match.append(((x, y), center_color_rgb.tolist()))
            print(f"원 위치: ({x}, {y}), 색상: {center_color_rgb}")

        groups = group_elements(Re_match)
        sorted_result = sort_groups(groups)

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

def adb_tap(x, y):
    os.system(f"adb shell input tap {x} {y}")
    
def watch_and_input(touch_range, d):
    while touch_range:
        coords_to_click = []

        # Step 1: Read touch_range.
        while touch_range:
            coord = touch_range[0]
            if isinstance(coord, tuple) and not math.isnan(coord[0]) and not math.isnan(coord[1]):
                coords_to_click.append((int(coord[0]), int(coord[1])))
                touch_range.pop(0)
            else:
                break

        # Step 2: If there are coordinates to click, click them in order.
        for x, y in coords_to_click:
            adb_tap(x, y)

        # Step 3: Clear the coords_to_click list.
        coords_to_click.clear()
        
        # Step 4: Wait for left ctrl click.
        while not keyboard.is_pressed('left ctrl'):
            time.sleep(0.05)

        # Step 5: If I click left ctrl, click next coordinates until get nan.
        coord = touch_range.pop(0)
        if touch_range:
            coord = touch_range[0]
            while isinstance(coord, tuple) and not math.isnan(coord[0]) and not math.isnan(coord[1]):
                coords_to_click.append((int(coord[0]), int(coord[1])))
                touch_range.pop(0)
                if touch_range:
                    coord = touch_range[0]
                else:
                    break
            for x, y in coords_to_click:
                adb_tap(x, y)
            coords_to_click.clear()
    
    # Step 6: If no data in touch_range, return True.
    return True

def color_distance(color1, color2, max_individual_difference=10, max_total_difference=20):
    individual_differences = [abs(c1 - c2) for c1, c2 in zip(color1, color2)]
    total_difference = sum(individual_differences)
    return all(d <= max_individual_difference for d in individual_differences) and total_difference <= max_total_difference

def find_matching_points(upper_points, lower_points, max_individual_difference=10, max_total_difference=20):
    matched_points = []
    matched_indices = set()
    
    for u_point in upper_points:
        u_coord, u_color = u_point
        match_found = False
        for i, l_point in enumerate(lower_points):
            l_coord, l_color = l_point
            if color_distance(u_color, l_color, max_individual_difference, max_total_difference):
                matched_points.append(l_coord)
                matched_indices.add(i)
                match_found = True
                break
        if not match_found:
            matched_points.append((math.nan, math.nan))

    return matched_points

def main():
    device_id = None  # 필요 시 특정 장치 ID로 변경
    d = connect_device(device_id)
    i = 0
    while True:
        triger = False
        image = capture_screenshot(d)
        upper_image, lower_image = extract_and_save_subimages(image)
        cv2.imwrite(f"./data/class1/upper_{i}.jpg", upper_image)
        cv2.imwrite(f"./data/class2/lower_{i}.jpg", lower_image)
        upper_touch_points = find_pattern(upper_image)
        if upper_touch_points is None:
            continue
        lower_touch_points = find_pattern(lower_image)
        if lower_touch_points is None:
            continue
        i += 1
        d.click(109,1965)
        time.sleep(0.1)
        d.click(1009,250)
        time.sleep(1)
        triger = True
        if triger:
            continue
        cv2.imwrite("./upper_touch_points.jpg", upper_image)
        cv2.imwrite("./lower_touch_points.jpg", lower_image)
        upper_adjusted = adjust_coordinates(upper_touch_points, 300)
        lower_adjusted = adjust_coordinates(lower_touch_points, 1400)
        # Match coordinates
        print("upper adjust")
        print(upper_adjusted)
        print("lower adjust")
        print(lower_adjusted)
        matched_points = find_matching_points(upper_adjusted, lower_adjusted)
        print("matched points")
        print(matched_points)
        cleaned_touch_range = remove_consecutive_nans(matched_points)

        print(cleaned_touch_range)
        if watch_and_input(cleaned_touch_range, d):
            continue  # 모두 'Nan'이어서 스크린샷을 다시 캡처
        

if __name__ == "__main__":
    main()
