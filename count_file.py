import os

def count_files(folder_path):
    # 폴더가 존재하지 않으면 0을 반환
    if not os.path.exists(folder_path):
        return 0

    # 폴더 내의 파일 개수 세기 (디렉토리는 제외)
    return sum(os.path.isfile(os.path.join(folder_path, f)) for f in os.listdir(folder_path))

# 세고 싶은 폴더 여기에 적어!!!!!!!
folder = 'after_prep_img_clean'
file_count = count_files(folder)
print(f"'{folder}' 폴더에 있는 파일 개수: {file_count}개")
