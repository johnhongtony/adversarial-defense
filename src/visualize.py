
# ============================================================
# Day 3. CIFAR-10 이미지 시각화 및 RGB 채널 분석
# ============================================================
# 목표:
# 1. CIFAR-10 이미지 16장 가져오기
# 2. 이미지와 정답 Label 시각화
# 3. 이미지의 Tensor Shape 및 픽셀값 범위 확인
# 4. RGB 채널별 평균 픽셀값 확인
# ============================================================

# 라이브러리 import
import torch
# torch: Tensor 연산, GPU 연산, 딥러닝 모델 구현
from torchvision import datasets, transforms
# datasets:	CIFAR-10 등 데이터셋 불러오기, transforms: 이미지 전처리
import matplotlib.pyplot as plt

# ① CIFAR-10 클래스 이름
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
# CIFAR-10은 총 10개의 클래스로 구성
# 각 클래스는 0~9까지의 정수 Label을 가짐
# 예: Label 0 = airplane, Label 3 = cat

# ② 이미지 전처리
transform = transforms.ToTensor()
# ToTensor()는 이미지 데이터를 PyTorch Tensor로 변환하는 함수
# 주요 작업 세 가지
# 1. 이미지 데이터를 Tensor로 변환
# 2. 이미지 차원을 HWC에서 CHW로 변경
# 3. 일반적인 uint8 이미지의 픽셀값을 0~255에서 0~1 범위로 변환
# HWC: Height(높이) × Width(너비) × Channel(채널)
# CHW: Channel(채널) × Height(높이) × Width(너비)
# CIFAR-10 이미지의 경우 변환 후 [32, 32, 3] -> [3, 32, 32] 형태가 됨

# ③ CIFAR-10 학습 데이터 불러오기
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
# 각 인자의 의미는 다음과 같음
# root="./data" : 데이터 저장 위치
# train=True: 학습 데이터 선택 (CIFAR-10 학습 데이터는 총 50,000장)
# download=True: 데이터가 없으면 다운로드
# transform=transform: 앞에서 정의한 이미지 전처리 적용
# train_dataset은 CIFAR-10 데이터를 관리하는 Dataset 객체
# 이미지와 정답 Label을 인덱스로 개별 접근할 수 있음

# ④ ★★★ 이미지 16장 가져오기
images = []
labels = []
# images: 가져온 이미지 Tensor를 저장할 빈 리스트
# labels: 각 이미지의 정답 Label을 저장할 빈 리스트

for i in range(16):
    image, label = train_dataset[i]
    # 이전 data_test.py 에서는 DataLoader를 통해 Batch 단위로 데이터를 가져왔으나,
    # 이번에는 직접 Dataset에 접근하여 데이터를 불러옴
    # train_dataset[i] -> 이미지 1장과 Label 1개
    # (이전 data_test.py) next(iter(train_loader)) -> 이미지 64장과 Label 64개
    # ∴ 이번에는 DataLoader를 사용하지 않아도 됌.
    images.append(image)
    labels.append(label)

# ⑤ 이미지 시각화
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):

    # ★★★ CHW → HWC 변환, Tensor의 차원을 CHW(0,1,2)에서 HWC(1,2,0)로 변경
    image = images[i].permute(1, 2, 0)
    # PyTorch와 Matplotlib은 이미지 차원을 다르게 취급하기 때문에 permute()를 사용하여 차원순서 재배열
    # CHW (Channels-First)(C, H, W)는 PyTorch 텐서 기본 형태, 모델 입력/출력
    # HWC (Channels-Last)(H, W, C)는 matplotlib.imshow()

    # 이미지 출력
    ax.imshow(image)

    # 클래스 이름 출력
    ax.set_title(classes[labels[i]])

    # 축 제거
    ax.axis("off")

plt.tight_layout()
plt.show()

# ⑥ 이미지별 RGB 채널 및 픽셀값 확인
for i in range(16):
    print(f"\nImage {i+1}")
    print("Label:", classes[labels[i]])
    print("Shape:", images[i].shape)
    print("Min:", images[i].min().item())
    print("Max:", images[i].max().item())

    print("R Channel Mean:", images[i][0].mean().item())
    print("G Channel Mean:", images[i][1].mean().item())
    print("B Channel Mean:", images[i][2].mean().item())