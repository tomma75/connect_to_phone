import math
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# 이미지 크기 설정
img_height, img_width = 224, 224

# 데이터 디렉토리 설정
class1_folder = 'data/class1'
class2_folder = 'data/class2'

# 이미지 로드 및 전처리
def load_and_preprocess_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=(img_height, img_width))
        img = img_to_array(img)
        img = preprocess_input(img)
        images.append(img)
        filenames.append(filename)
    return np.array(images), filenames

# 색상 추출 함수 (K-means를 활용한 주요 색상 추출)
def extract_dominant_color(image, k=1, n_init=10):
    pixels = np.float32(image.reshape(-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=n_init)
    kmeans.fit(pixels)
    palette = kmeans.cluster_centers_
    return palette

# 패턴 찾기 (원 검출)
def find_pattern(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("유효하지 않은 이미지 형식입니다. NumPy 배열이어야 합니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    circles = cv2.HoughCircles(morph, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=30, maxRadius=70)
    Re_match = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            center_color_bgr = image[y, x]
            center_color_rgb = center_color_bgr[::-1]  # Convert BGR to RGB format
            if center_color_rgb[0] == center_color_rgb[1] == center_color_rgb[2]:
                continue
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
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
# 색상 거리 계산
def color_distance(color1, color2, max_individual_difference=10, max_total_difference=20):
    individual_differences = [abs(c1 - c2) for c1, c2 in zip(color1, color2)]
    total_difference = sum(individual_differences)
    return all(d <= max_individual_difference for d in individual_differences) and total_difference <= max_total_difference

# 매칭 포인트 찾기
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

# 좌표 예측 함수
def predict_and_match_coordinates(model, class1_colors, class2_images):
    predictions = model.predict(class1_colors[:, 0])
    matched_indices = np.argmax(predictions, axis=1)
    
    coordinates = []
    for idx in matched_indices:
        matched_image = class2_images[idx]
        
        # 이미지를 그레이스케일로 변환
        matched_image_gray = cv2.cvtColor(matched_image, cv2.COLOR_BGR2GRAY)
        
        # 이진화 이미지 생성
        _, thresholded = cv2.threshold(matched_image_gray, 1, 255, cv2.THRESH_BINARY)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 첫 번째 윤곽선의 중심 계산
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                coordinates.append((cX, cY))
            else:
                coordinates.append((0, 0))
        else:
            coordinates.append((0, 0))
    
    return coordinates

# 피드백 수집 및 라벨링 함수
def collect_feedback(coordinates):
    labels = []
    for coord in coordinates:
        print(f"Predicted coordinate: {coord}")
        feedback = input("Is this correct? (O/X): ")
        labels.append(1 if feedback.lower() == 'o' else 0)
    return np.array(labels)

# 학습 데이터 수집 및 모델 학습
def train_with_feedback(model, class1_colors, class2_images):
    while True:
        coordinates = predict_and_match_coordinates(model, class1_colors, class2_images)
        labels = collect_feedback(coordinates)
        
        # 라벨링된 데이터를 사용하여 모델 학습
        class1_labels = tf.keras.utils.to_categorical(labels, num_classes=class2_images.shape[0])
        model.fit(class1_colors[:, 0], class1_labels, epochs=1, batch_size=4)
        
        # 모델 저장
        model.save('color_matching_model.h5')

# 주 함수
def main():
    # 이미지 로드 및 전처리
    class1_images, class1_filenames = load_and_preprocess_images(class1_folder)
    class2_images, class2_filenames = load_and_preprocess_images(class2_folder)

    # Class1과 Class2에서 주요 색상 추출
    class1_colors = np.array([extract_dominant_color(img, k=1) for img in class1_images])
    class2_colors = np.array([extract_dominant_color(img, k=1) for img in class2_images])

    # 모델 구성
    model = Sequential([
        Input(shape=(3,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(class2_colors.shape[0], activation='softmax')  # class2 이미지 수만큼의 출력 노드
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # 학습 실행
    train_with_feedback(model, class1_colors, class2_images)

if __name__ == "__main__":
    main()
