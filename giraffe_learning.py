# vscode 에서 실행하는 방법
# 1. venv 설정하고 venv로 powershell 실행하기
# 2. 필요한 라이브러리 깔기
# 3. stylegan3 clone하고 cd stylegan3
# 4. 사용할 이미지 폴더 리사이징, 전처리 후 zip으로 압축
# 5. !python train.py \
#  --outdir=training-runs \
#  --cfg=stylegan3-t \
#  --data=/content/my_dataset.zip \
#  --gpus=1 \
#  --batch=4 \
#  --gamma=8.2 \
#  --mirror=1 \
#  --aug=noaug \
#  --snap=10 \
#  --kimg=200 \
#--metrics=none













'''
!git clone https://github.com/NVlabs/stylegan3.git

from google.colab import drive
drive.mount('/content/drive')
%cd /content/stylegan3

# ✅ Python 3.10 설치 (torchvision 0.14.1를 지원하는 python 버전을 설치하는 과정 )
!sudo apt-get update
!sudo apt-get install python3.10 python3.10-distutils python3.10-dev -y

# ✅ Python 3.10을 기본 python3으로 설정
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
!sudo update-alternatives --config python3

# ✅ get-pip.py를 정확한 주소에서 다운로드
!curl -O https://bootstrap.pypa.io/get-pip.py

# ✅ Python 3.10으로 pip 설치
!python3 get-pip.py

# ✅ pip 업그레이드 (선택)
!python3 -m pip install --upgrade pip setuptools

# ✅ PyTorch 1.13.1 + torchvision 0.14.1 (CUDA 11.6)
!pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116



# 기린데이터셋 업로드용
!git clone https://github.com/HGM0223/2025AI_Giraffe

# 필요한 라이브러리 설치
!pip install ninja imageio-ffmpeg
!pip install click requests tqdm pyspng

# 기린 데이터셋 리사이징

from PIL import Image
import os

input_folder = "/content/stylegan3/2025AI_Giraffe/giraffe_photo"   # 원본 이미지 폴더
output_folder = "/content/stylegan3/2025AI_Giraffe/giraffe_photo"  # 리사이징된 이미지 저장 폴더
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = Image.open(os.path.join(input_folder, filename)).convert("RGB")
        img = img.resize((1024, 1024), Image.LANCZOS)
        img.save(os.path.join(output_folder, filename))

# 기린 데이터셋 zip압축 (gan3에선 학습할 때 zip파일을 사용함)
!zip -r my_dataset.zip /content/stylegan3/2025AI_Giraffe/giraffe_photo



# 학습시 오류날 경우 사용
!pip install psutil
!pip install numpy==1.26.4
!pip install scipy


# 학습 시작
!python train.py \
  --outdir=training-runs \
  --cfg=stylegan3-t \
  --data=/content/my_dataset.zip \
  --gpus=1 \
  --batch=4 \
  --gamma=8.2 \
  --mirror=1 \
  --aug=noaug \
  --snap=10 \
  --kimg=200 \
--metrics=none


'''