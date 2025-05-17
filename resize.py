import os
from PIL import Image

import cv2
import shutil

#이미지 화질 고정
input4pixel = '../giraffe_photo'
output4pixel = '../after_prep_img'

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
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                save_path = os.path.join(output4pixel, filename)
                img.save(save_path)
        except Exception as e: 
            print(f"Error processing {filename}: {e}")
            continue



