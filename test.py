import re

# 주어진 RGB 값들을 기준으로 하는 목록
reference_rgb_values = [
    (164, 190, 142), (59, 94, 156), (104, 140, 203),
    (87, 122, 184), (121, 154, 213), (90, 128, 184), (97, 136, 196)
]

def euclidean_distance(rgb1, rgb2):
    """두 RGB 값 간의 유클리드 거리 계산"""
    return sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)) ** 0.5

def extract_similar_rgb_from_file(file_path, reference_values, threshold=30):
    similar_rgb_values = []
    
    # RGB 값을 추출하기 위한 정규 표현식 패턴
    rgb_pattern = r"\((\d+), (\d+), (\d+)\)"

    with open(file_path, 'r') as file:
        for line in file:
            # RGB 값 추출
            match = re.search(rgb_pattern, line)
            if match:
                rgb = tuple(map(int, match.groups()))

                # 각 참조 값과의 유클리드 거리 계산
                for ref_rgb in reference_values:
                    distance = euclidean_distance(rgb, ref_rgb)
                    if distance <= threshold:
                        similar_rgb_values.append(rgb)
                        break

    return similar_rgb_values

def save_rgb_to_file(rgb_values, output_path):
    with open(output_path, 'w') as file:
        for rgb in rgb_values:
            file.write(f"{rgb}\n")

# 파일 경로 설정
input_file_path = './log.txt'  # 텍스트 파일 경로
output_file_path = './log_blue.txt'  # 저장할 출력 파일 경로

# 유사한 RGB 값 추출 및 저장
similar_rgb_values = extract_similar_rgb_from_file(input_file_path, reference_rgb_values, threshold=30)
save_rgb_to_file(similar_rgb_values, output_file_path)

print(f"유사한 RGB 값들이 {output_file_path}에 저장되었습니다.")
