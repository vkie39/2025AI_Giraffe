import os
from collections import defaultdict

def check_duplicate_filenames(classified_root='classified'):
    """
    left, right, front 폴더 내부에서
    같은 파일 이름이 2개 이상 폴더에 존재하는 경우를 찾아줌
    """
    file_tracker = defaultdict(set)  # {파일이름: {폴더1, 폴더2, ...}}

    for dir_name in ['left', 'right', 'front']:
        dir_path = os.path.join(classified_root, dir_name)
        if not os.path.exists(dir_path):
            continue
        for file in os.listdir(dir_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_tracker[file].add(dir_name)

    # 중복 확인
    duplicates = {name: folders for name, folders in file_tracker.items() if len(folders) > 1}

    if duplicates:
        print("🚨 중복된 파일명 발견됨:")
        for name, folders in duplicates.items():
            print(f"- {name} → 존재 폴더: {', '.join(sorted(folders))}")
    else:
        print("✅ 중복된 파일 없음!")

if __name__ == "__main__":
    check_duplicate_filenames('classified')
