import os
from PIL import Image

import cv2
import shutil

import dlib
import numpy as np

#이미지 화질 고정
input4pixel = 'merged_output'  # 원본 이미지 폴더
output4pixel = 'after_prep_img'

os.makedirs(output4pixel, exist_ok=True)

image_extensions = ['.jpg', '.jpeg', '.png']

target_size = (512, 512)

#이미지 전처리 반복
for filename in os.listdir(input4pixel):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        img_path = os.path.join(input4pixel, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = img.resize(target_size, Image.Resampling.LANCZOS) #고화질 resize
                save_path = os.path.join(output4pixel, filename)
                img.save(save_path)
        except Exception as e: 
            print(f"Error processing {filename}: {e}")
            continue

#흐린 이미지 제거(라플라시안 이용)

def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold # variance가 threshold보다 작으면 흐린 이미지로 판단

input4blur = 'after_prep_img'        # 원본 이미지 폴더
output4blur = 'filtered_images'    # 선명한 이미지 저장 폴더
blurry_folder = 'blurry_images'      # 흐린 이미지 저장 폴더

# 출력 폴더가 없으면 생성
os.makedirs(output4blur, exist_ok=True)
os.makedirs(output4blur, exist_ok=True)

# 대상 디렉터리가 존재하는지 확인하고, 없으면 생성
destination_dir = os.path.dirname(blurry_folder)
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

threshold = 150.0  # 블러 판단 기준 (일반적으로 50~150 사이로 조정)


for filename in os.listdir(input4blur):
    file_path = os.path.join(input4blur, filename)
    image = cv2.imread(file_path)

    if image is None:
        print(f"이미지를 읽을 수 없습니다: {filename}")
        continue

    if is_blurry(image, threshold):
        print(f"[흐림] {filename}")
        shutil.move(file_path, os.path.join(blurry_folder, filename))
    else:
        print(f"[선명] {filename}")
        shutil.move(file_path, os.path.join(output4blur, filename))

print("블러 처리 완료!")

#사람 얼굴 데이터셋
'''
# 측면 얼굴 제거 코드(양쪽 눈이 안 보이면 이미지 다른 폴더로 이동)
input4EYE = 'filtered_images'  # 선명한 이미지 폴더
output4EYE = 'cropped_faces'   # 크롭된 이미지 저장 폴더
side_face_folder = 'side_faces' # 측면 얼굴 이미지 저장 폴더

os.makedirs(input4EYE, exist_ok=True)
os.makedirs(side_face_folder, exist_ok=True)

# dlib의 얼굴 검출기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def align_face(image, landmarks):
    # 왼쪽 눈과 오른쪽 눈의 중심 좌표 계산
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)

    # 눈 사이의 각도 계산
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    # 이미지 중심 좌표 계산
    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)

    # 회전 행렬 생성
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)

    # 이미지 회전
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_CUBIC)

    return rotated

def is_eye_visible(landmarks, eye_indices):
    # 눈 영역의 좌표 추출
    eye_points = landmarks[eye_indices]
    # 눈 영역의 너비와 높이 계산
    x_coords = eye_points[:, 0]
    y_coords = eye_points[:, 1]
    width = x_coords.max() - x_coords.min()
    height = y_coords.max() - y_coords.min()
    # 눈 영역의 크기가 일정 기준 이상이면 눈이 보인다고 판단
    return width > 5 and height > 5

image_extensions = ['.jpg', '.jpeg', '.png']

for filename in os.listdir(input4EYE):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        img_path = os.path.join(input4EYE, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"이미지를 읽을 수 없습니다: {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            print(f"얼굴을 찾을 수 없습니다: {filename}")
            continue

        for rect in faces:
            shape = predictor(gray, rect)
            landmarks = np.array


#얼굴 중심으로 사진 크롭(랜드마크 매핑 이용)

input4crop = 'human_face'  # 선명한 이미지 폴더
output4crop = 'cropped_faces'   # 크롭된 이미지 저장 폴더

os.makedirs(output4crop, exist_ok=True)


# dlib의 얼굴 검출기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def align_face(image, landmarks):
    # 왼쪽 눈과 오른쪽 눈의 중심 좌표 계산
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)

    # 눈 사이의 각도 계산
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    # 이미지 중심 좌표 계산
    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)

    # 회전 행렬 생성
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)

    # 이미지 회전
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_CUBIC)

    return rotated

for filename in os.listdir(input4crop):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        img_path = os.path.join(input4crop, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"이미지를 읽을 수 없습니다: {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            print(f"얼굴을 찾을 수 없습니다: {filename}")
            continue

        for rect in faces:
            shape = predictor(gray, rect)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            # 얼굴 정렬
            aligned_face = align_face(image, landmarks)

            # 정렬된 이미지에서 얼굴 영역 추출
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

            # 사이즈 조정: 상하좌우로 20픽셀 여유를 줌
            margin = 20
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, aligned_face.shape[1])
            y2 = min(y + h + margin, aligned_face.shape[0])

            cropped_face = aligned_face[y1:y2, x1:x2]

            # 크롭된 얼굴 이미지 저장
            save_path = os.path.join(output4crop, filename)
            cv2.imwrite(save_path, cropped_face)
            print(f"크롭된 얼굴 저장 완료: {save_path}")

#기린 얼굴 데이터셋

'''