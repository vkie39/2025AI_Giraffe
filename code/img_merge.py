import os
import shutil
from PIL import Image

# 두 폴더 속 사진들을 병합하여 새로 이름을 붙이고 merged_output으로 저장

def rename_and_merge_images(input_dirs, output_dir, target_ext=".jpg"):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ['.jpg', '.jpeg', '.png']
    index = 1

    for input_dir in input_dirs:
        for filename in sorted(os.listdir(input_dir)):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                src_path = os.path.join(input_dir, filename)
                dst_path = os.path.join(output_dir, f"{index}{target_ext}")
                try:
                    # 이미지 열어서 RGB로 저장 (포맷 통일)
                    with Image.open(src_path) as img:
                        img = img.convert("RGB")
                        img.save(dst_path)
                    print(f"Saved: {dst_path}")
                    index += 1
                except Exception as e:
                    print(f"[ERROR] {filename} 처리 중 오류 발생: {e}")

    print("✅ 이미지 병합 및 이름 재지정 완료!")


rename_and_merge_images(['giraffe_photo', 'giraffe_face2'], 'merged_output')
