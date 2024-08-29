import cv2
import numpy as np

def colors_are_similar(color1, color2, tolerance):
    return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2))

def circle_find():
    # 색상 비교 시 허용 오차 설정
    color_tolerance = 30
    image_path_1 = "./1.png"
    image_path_2 = "./2.png"
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)

    # 이미지를 HSV 색상 공간으로 변환
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # 가우시안 블러 적용
    blurred1 = cv2.GaussianBlur(hsv1, (15, 15), 0)
    blurred2 = cv2.GaussianBlur(hsv2, (15, 15), 0)

    # 원 검출
    circles1 = cv2.HoughCircles(blurred1[:, :, 2], cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                param1=50, param2=30, minRadius=20, maxRadius=50)
    circles2 = cv2.HoughCircles(blurred2[:, :, 2], cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                param1=50, param2=30, minRadius=20, maxRadius=50)

    # 원이 검출되었는지 확인
    detected_circles1 = []
    detected_circles2 = []
    if circles1 is not None:
        circles1 = np.round(circles1[0, :]).astype("int")
        for (x, y, r) in circles1:
            # 원 중심 및 주변의 색상 평균화
            mask1 = np.zeros_like(hsv1[:, :, 2])
            cv2.circle(mask1, (x, y), r, (255, 255, 255), -1)
            mean_color1 = cv2.mean(hsv1, mask=mask1)[:3]  # HSV 평균 추출
            mean_color1 = tuple(map(int, mean_color1))
            
            detected_circles1.append((x, y, r, mean_color1))

    if circles2 is not None:
        circles2 = np.round(circles2[0, :]).astype("int")
        for (x, y, r) in circles2:
            # 원 중심 및 주변의 색상 평균화
            mask2 = np.zeros_like(hsv2[:, :, 2])
            cv2.circle(mask2, (x, y), r, (255, 255, 255), -1)
            mean_color2 = cv2.mean(hsv2, mask=mask2)[:3]  # HSV 평균 추출
            mean_color2 = tuple(map(int, mean_color2))
            
            detected_circles2.append((x, y, r, mean_color2))

    # 첫 번째 이미지와 두 번째 이미지의 원 중심 색상 비교하여 입력 순서 결정
    input_sequence = []
    for circle2 in detected_circles2:
        color2 = circle2[3]
        for i, circle1 in enumerate(detected_circles1):
            color1 = circle1[3]
            if colors_are_similar(color2, color1, color_tolerance):
                input_sequence.append(i + 1)
                break
    return input_sequence, detected_circles1, detected_circles2

input_sequence, detected_circles1, detected_circles2 = circle_find()
print(input_sequence, detected_circles1, detected_circles2)
