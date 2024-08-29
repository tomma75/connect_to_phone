import os
import subprocess
import cv2
import numpy as np
import time

def capture_screenshot_to_memory(device_id=None):
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
    # Split the image into upper and lower parts at y=1400
    upper_image = image[300:1400, :]
    lower_image = image[1400:2230, :]

    # Save upper and lower images
    cv2.imwrite("1.png", upper_image)
    cv2.imwrite("2.png", lower_image)

    # Convert lower image to HSV color space
    hsv = cv2.cvtColor(lower_image, cv2.COLOR_BGR2HSV)

    # Define range of colors in HSV
    color_ranges = [
        ((0, 70, 50), (10, 255, 255)),     # Red range
        ((20, 100, 100), (30, 255, 255)),  # Yellow range
        ((110, 100, 100), (130, 255, 255)), # Purple range
        ((160, 70, 50), (180, 255, 255))   # Another Red range
    ]

    # Apply masks and find contours for each color range
    buttons = []
    for (lower, upper) in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            if len(approx) > 8:  # Approximate to a circle
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                if radius > 10:  # Filter out small circles
                    button_img = lower_image[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
                    if button_img.shape[0] > 0 and button_img.shape[1] > 0:  # Check if the slice is valid
                        buttons.append(button_img)

    # Save the extracted button images
    for i, button_img in enumerate(buttons):
        cv2.imwrite(f"2-{i+1}.png", button_img)

def load_template_as_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()

def decode_template(image_bytes):
    np_array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

def analyze_screenshot(image, template, threshold=0.8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return loc, w, h

def tap(x, y, device_id=None):
    adb_command = ["adb"]
    if device_id:
        adb_command += ["-s", device_id]
    adb_command += ["shell", "input", "tap", str(x), str(y)]
    subprocess.run(adb_command)

def detect_blur(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def main(device_id=None):
    while True:
        image = capture_screenshot_to_memory(device_id)

        # Extract and save sub-images
        extract_and_save_subimages(image)

        # Load templates as bytes
        template_1_bytes = load_template_as_bytes("1.png")
        template_2_1_bytes = load_template_as_bytes("2-1.png")
        template_2_2_bytes = load_template_as_bytes("2-2.png")
        template_2_3_bytes = load_template_as_bytes("2-3.png")
        template_2_4_bytes = load_template_as_bytes("2-4.png")

        # Decode templates from bytes
        template_1 = decode_template(template_1_bytes)
        template_2_1 = decode_template(template_2_1_bytes)
        template_2_2 = decode_template(template_2_2_bytes)
        template_2_3 = decode_template(template_2_3_bytes)
        template_2_4 = decode_template(template_2_4_bytes)

        # Step 2: Get input order from 1.png
        loc_1, w_1, h_1 = analyze_screenshot(image, template_1)
        input_order = [(pt[0] + w_1 // 2, pt[1] + h_1 // 2) for pt in zip(*loc_1[::-1])]

        # Step 3: Monitor non-input targets
        non_input_images = [template_2_1, template_2_2, template_2_3, template_2_4]
        non_input_locs = []

        for template_2 in non_input_images:
            loc_2, w_2, h_2 = analyze_screenshot(image, template_2)
            non_input_locs.extend([(pt[0] + w_2 // 2, pt[1] + h_2 // 2) for pt in zip(*loc_2[::-1])])

        # Step 4: Check if non-input target is blurred
        while True:
            image = capture_screenshot_to_memory(device_id)
            if all(detect_blur(image[y-h_2//2:y+h_2//2, x-w_2//2:x+w_2//2]) for x, y in non_input_locs):
                for x, y in input_order:
                    tap(x, y, device_id)
                    time.sleep(0.1)
                break

        # Step 5: Wait until non-input target changes
        while True:
            image = capture_screenshot_to_memory(device_id)
            if not all(detect_blur(image[y-h_2//2:y+h_2//2, x-w_2//2:x+w_2//2]) for x, y in non_input_locs):
                break

# Example usage
if __name__ == "__main__":
    main()
