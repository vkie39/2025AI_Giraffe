import os
from PIL import Image

import cv2
import shutil

#이미지 화질 고정
input_folder = 'giraffe_photo'
output_folder = 'after_prep_img'

os.makedirs(output_folder, exist_ok=True)

image_extensions = ['.jpg', '.jpeg', '.png']

target_size = (512, 512)

#이미지 전처리 반복
for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        img_path = os.path.join(input_folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img = img.resize(target_size, Image.ANTIALIAS) #고화질 resize
                save_path = os.path.join(output_folder, filename)
                img.save(save_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

#흐린 이미지 제거(라플라시안 이용)

#

#사람 얼굴 데이터셋

#랜드마크 매핑


#기린 얼굴 데이터셋

