import os
from PIL import Image
import imagehash

folder_path = "your_folder_path"  # 사진 폴더 경로

hashes = set()

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        file_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(file_path)
            img_hash = imagehash.phash(img)
            # 해시값이 100% 같을 때만 중복으로 판단
            if img_hash in hashes:
                print(f"중복 삭제: {file_path}")
                os.remove(file_path)
            else:
                hashes.add(img_hash)
        except Exception as e:
            print(f"에러: {file_path} - {e}")
