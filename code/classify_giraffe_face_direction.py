import os
import cv2
import numpy as np
import shutil

def classify_giraffe_face_direction(image_path):
    """기린 이미지 한 장의 얼굴 방향(left, right, front) 분류"""
    img = cv2.imread(image_path)
    if img is None:
        return 'unreadable'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    left_half = thresh[:, :w // 2]
    right_half = thresh[:, w // 2:]

    left_intensity = np.sum(left_half)
    right_intensity = np.sum(right_half)

    ratio = left_intensity / (right_intensity + 1e-5)

    if ratio > 1.3:
        return 'left'
    elif ratio < 0.7:
        return 'right'
    else:
        return 'front'

def ensure_folder(path):
    """폴더가 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)

def classify_and_move_images(source_folder, output_root='classified'):
    """이미지를 방향별로 분류하고 폴더에 저장"""
    directions = ['left', 'right', 'front', 'unreadable']

    # 폴더 생성
    for d in directions:
        ensure_folder(os.path.join(output_root, d))

    for file_name in sorted(os.listdir(source_folder)):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(source_folder, file_name)
            direction = classify_giraffe_face_direction(image_path)
            destination = os.path.join(output_root, direction, file_name)
            shutil.copy(image_path, destination)
            print(f"{file_name} → {direction}")

def main():
    source_folder = 'after_prep_img_clean'          # 원본 이미지 폴더
    output_root = 'classified'        # 결과 저장 폴더
    classify_and_move_images(source_folder, output_root)

if __name__ == "__main__":
    main()