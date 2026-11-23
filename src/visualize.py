
# ============================================================
# Day 3. CIFAR-10 이미지 시각화 및 RGB 채널 분석
# ============================================================
# 목표:
# 1. CIFAR-10 이미지 16장 가져오기
# 2. 이미지와 정답 Label 시각화
# 3. 이미지의 Tensor Shape 및 픽셀값 범위 확인
# 4. RGB 채널별 평균 픽셀값 확인
# ============================================================


# 1. 라이브러리 import

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# torch: Tensor 연산 및 딥러닝 모델 구현
# datasets: CIFAR-10 등의 데이터셋 불러오기
# transforms: 이미지 전처리
# matplotlib.pyplot: 이미지 및 그래프 시각화


# 2. CIFAR-10 클래스 이름 정의

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CIFAR-10은 총 10개의 클래스로 구성
# Label 번호 0~9가 각 클래스 이름에 대응


# 3. 이미지 전처리 설정

transform = transforms.ToTensor()

# ToTensor():
# 1. 이미지 데이터를 PyTorch Tensor로 변환
# 2. 이미지 차원을 HWC → CHW로 변경
# 3. 일반적인 uint8 이미지의 픽셀값을 0~255 → 0~1로 정규화
#
# HWC: Height × Width × Channel
# CHW: Channel × Height × Width
# CIFAR-10 이미지 변환 후 Shape: [3, 32, 32]


# 4. CIFAR-10 학습 데이터 불러오기

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# root: 데이터셋 저장 경로
# train=True: 학습 데이터 선택 (50,000장)
# download=True: 데이터셋이 없으면 다운로드
# transform: 앞에서 설정한 이미지 전처리 적용


# 5. Dataset에서 이미지 16장 가져오기

images = []
labels = []

for i in range(16):
    image, label = train_dataset[i]
    # ★ train_dataset[] PyTorch의 Dataset 객체로써 
    # 파이썬에서 객체에 [i]처럼 대괄호 인덱싱을 하면 
    # 내부적으로 __getitem__(i)라는 약속된 메서드가 실행
    #
    # datasets.CIFAR10의 __getitem__(i)는 i번째 데이터를 찾아 
    # 다음 두 가지 값을 한 쌍(튜플, (이미지_텐서, 정답_정수))으로 돌려줌
    # ∴ i번째 이미지와 그 이미지에 대응하는 정답 Label을 반환

    images.append(image)
    labels.append(label)

# Dataset은 인덱스를 이용해 개별 데이터에 직접 접근 가능
# train_dataset[i] → 이미지 1장(Tensor) + 정답 Label 1개
#
# DataLoader를 사용했던 이전 실습(data_test.py)과 달리 
# 이번에는 이미지 16장을 개별적으로 가져옴
#
# images: 이미지 Tensor 16개 저장
# labels: 각 이미지의 정답 Label 16개 저장
#
# 이미지 1장의 Shape: [3, 32, 32]
# Batch 단위가 아니므로 별도의 DataLoader는 사용하지 않음


# 6. 이미지 16장 시각화

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

# 4행 × 4열의 이미지 출력 영역 생성
# fig: 전체 그림을 관리하는 Figure 객체
# axes: 개별 이미지 출력 영역인 Axes 객체 배열
# figsize: 전체 그림의 크기 (가로 10인치 × 세로 10인치)


for i, ax in enumerate(axes.flat):

    # PyTorch(CHW)와 Matplotlib(HWC)의 이미지 차원 순서가 다르므로 변경
    image = images[i].permute(1, 2, 0)

    # 기존 Shape: [3, 32, 32] (C, H, W)
    # 변경 Shape: [32, 32, 3] (H, W, C)
    # permute(1, 2, 0): 기존 차원의 순서를 H → W → C로 재배열

    ax.imshow(image)
    # imshow(): RGB 이미지 데이터를 해당 Axes 영역에 출력

    ax.set_title(classes[labels[i]])
    # 해당 이미지의 정답 Label을 클래스 이름으로 변환하여 제목으로 표시

    ax.axis("off")
    # 이미지 주변의 X축, Y축 및 눈금 제거


plt.tight_layout()
# 이미지 사이의 간격을 자동 조정하여 제목이나 이미지가 겹치지 않도록 설정

plt.show()
# 최종 이미지 시각화


# 7. 이미지별 Tensor 및 픽셀값 분석

for i in range(16):

    print(f"\nImage {i+1}")

    # 이미지 기본 정보
    print("Label:", classes[labels[i]])
    # 정답 Label을 클래스 이름으로 변환하여 출력

    print("Shape:", images[i].shape)
    # 이미지 Tensor의 차원 확인
    # [3, 32, 32] = [RGB Channel, Height, Width]

    # 전체 RGB 픽셀값 범위 확인
    print("Min:", images[i].min().item())
    print("Max:", images[i].max().item())

    # min(): 이미지 전체 픽셀값 중 최솟값
    # max(): 이미지 전체 픽셀값 중 최댓값
    # item(): Tensor의 단일 값을 Python 숫자로 변환
    # ToTensor() 적용 후 픽셀값 범위는 0~1

    # RGB 채널별 평균 픽셀값 확인
    print("R Channel Mean:", images[i][0].mean().item())
    print("G Channel Mean:", images[i][1].mean().item())
    print("B Channel Mean:", images[i][2].mean().item())

    # images[i][0]: R(Red) 채널 → Shape [32, 32]
    # images[i][1]: G(Green) 채널 → Shape [32, 32]
    # images[i][2]: B(Blue) 채널 → Shape [32, 32]
    #
    # mean(): 각 채널의 1,024개 픽셀값 평균 계산
    # 채널별 평균값을 비교하여 이미지의 전반적인 색상 분포 확인
    # 단, 평균값만으로 개별 픽셀의 색상이나 이미지 전체의 색상을 판단할 수는 없음
