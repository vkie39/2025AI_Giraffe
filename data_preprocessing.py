import os

from PIL import Image

import cv2
import shutil

#이미지 화질 고정
input4pixel = 'giraffe_photo'
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
                img = img.resize(target_size, Image.ANTIALIAS) #고화질 resize
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

input4blur = 'input_images'        # 원본 이미지 폴더
output4blur = 'filtered_images'    # 선명한 이미지 저장 폴더
blurry_folder = 'blurry_images'      # 흐린 이미지 저장 폴더

# 출력 폴더가 없으면 생성
os.makedirs(output4blur, exist_ok=True)
os.makedirs(output4blur, exist_ok=True)

threshold = 100.0  # 블러 판단 기준 (일반적으로 50~150 사이로 조정)


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

#얼굴 중심으로 사진 크롭(랜드마크 매핑 이용)


#기린 얼굴 데이터셋

