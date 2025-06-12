import os
from PIL import Image
import imagehash

# 폴더 경로를 지정하세요
folder1 = "giraffe0612_resized"
folder2 = "after_prep_img"

# 1번 폴더의 해시값 저장
hashes = set()
for filename in os.listdir(folder1):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        file_path = os.path.join(folder1, filename)
        try:
            img = Image.open(file_path)
            img_hash = imagehash.phash(img)
            hashes.add(img_hash)
        except Exception as e:
            print(f"에러(1): {file_path} - {e}")

# 2번 폴더와 1번 폴더 비교해서 중복이면 2번 폴더 사진 삭제
for filename in os.listdir(folder2):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        file_path = os.path.join(folder2, filename)
        try:
            img = Image.open(file_path)
            img_hash = imagehash.phash(img)
            if img_hash in hashes:
                print(f"중복 삭제: {file_path}")
                os.remove(file_path)
        except Exception as e:
            print(f"에러(2): {file_path} - {e}")
