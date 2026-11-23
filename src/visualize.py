
# ============================================================
# Day 3. CIFAR-10 이미지 시각화 및 RGB 채널 분석
# ============================================================
# 목표: 입력데이터를 이해한다.
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
# 1. 이미지 데이터 또는 Numpy 배열을 PyTorch Tensor로 변환
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
# 
# images.append(image)와 labels.append(label)의 역할
# .append(값)은 기존 리스트의 맨 끝에 새 요소를 추가하는 메서드
# 즉, 반복이 1바퀴(i = 0) 돌 때마다
# images는 [텐서0] → 2바퀴째엔 [텐서0, 텐서1] → ... → 16바퀴째엔 길이 16짜리 리스트가 됨
# labels도 마찬가지로 [라벨0, 라벨1, ..., 라벨15]인 길이 16짜리 리스트가 됨
# 



# 6. 이미지 16장 시각화
# 텐서 형태로 저장된 16장의 이미지를 사람이 눈으로 볼 수 있게 
# 4×4 격자판에 그려주는 시각화 단계

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

# 4행 × 4열의 이미지 출력 영역 생성
# fig(figure): 그림 전체가 그려지는 커다란 흰색 도화지 한 장, 전체 그림을 관리하는 Figure 객체
# axes: 개별 이미지 출력 영역인 Axes 객체 배열(도화지 안에 세로 4줄 × 가로 4줄로 나눈 16개의 작은 액자(서브플롯) 모음)
# figsize: 전체 그림의 크기 (가로 10인치 × 세로 10인치), 크기를 안 키우면 16개 그림과 제목이 서로 좁게 붙어서 글자가 겹침

# 2차원 액자 배열을 1줄로 펴서 순회하기
for i, ax in enumerate(axes.flat):
# axes.flat을 쓰는 이유:
# axes는 4행 4열의 2차원 구조라서 원래는 axes[0][0], axes[0][1]처럼 행과 열 번호를 두 번 적어야 접근할 수 있음
# .flat을 붙이면 4×4 칸을 첫 번째 줄 왼쪽부터 마지막 줄 오른쪽까지 0번~15번의 1줄(1차원)로 쭉 펴줌
# 덕분에 for문 하나로 16칸을 차례대로 돌 수 있음    
#
# enumerate(...)의 역할:
# axes.flat에서 액자(ax)를 하나씩 꺼낼 때 현재 몇 번째 칸인지 번호(i = 0, 1, ..., 15)까지 함께 붙여줌
# 이 i 덕분에 images[i]와 labels[i]에서 딱 맞는 이미지와 정답을 꺼낼 수 있음

    # PyTorch(CHW)와 Matplotlib(HWC)의 이미지 차원 순서가 다르므로 변경
    image = images[i].permute(1, 2, 0)

    # 기존 Shape: [3, 32, 32] (C, H, W)
    # 변경 Shape: [32, 32, 3] (H, W, C)
    # permute(1, 2, 0): 기존 차원의 순서를 H → W → C로 재배열

    ax.imshow(image)
    # imshow(): RGB 이미지 데이터를 해당 Axes 영역에 출력

    ax.set_title(classes[labels[i]])
    # 해당 이미지의 정답 Label을 클래스 이름으로 변환하여 제목으로 표시
    # labels[i]에는 숫자(예: 6)가 들어 있음
    # classes[6]을 조회해 사람이 읽을 수 있는 영어 단어("frog")로 바꾼 뒤 액자 위쪽 제목으로 달아줌

    ax.axis("off")
    # 이미지 주변의 X축, Y축 및 눈금 제거


plt.tight_layout()
# 이미지 사이의 간격을 자동 조정하여 제목이나 이미지가 겹치지 않도록 설정

plt.show()
# 최종 이미지 시각화
# 지금까지 메모리에 차곡차곡 그려둔 도화지(fig)를 실제 화면 창에 띄움


# 7. 이미지별 Tensor 및 픽셀값 분석
# 눈으로 본 이미지가 컴퓨터 내부에서는 어떤 숫자와 차원 구조로 이루어져 있는지 해부해 보는 수치 분석 단계
# ★ Adversarial Attack은 이미지의 픽셀값을 변형하는 방식으로 이루어질 확률이 높음

for i in range(16):
    # 앞선 시각화 코드에서는 enumerate(axes.flat)을 사용했는데,
    # 이번에는 단순히 이미지 데이터를 순서대로 출력하기 때문에 range(16)을 사용

    print(f"\nImage {i+1}")
    # 여기서는 Python의 f-string을 사용
    # f: 문자열 안에 변수를 삽입할 수 있도록 함
    # \n: 줄바꿈
    # {i+1}: 변수 i에 1을 더한 값을 출력
    # i+1 사용이유: Python의 인덱스는 0부터 시작하지만 사람이 확인할 때는 1부터 시작하는 것이 편하기 때문
    # i = 0 → Image 1
    # i = 1 → Image 2
    # ...
    # i = 15 → Image 16

    # 이미지 기본 정보
    print("Label:", classes[labels[i]])
    # 정답 Label을 클래스 이름으로 변환하여 출력
    # 예를 들어 Label이 7이라면 horse가 출력

    print("Shape:", images[i].shape)
    # 이미지 Tensor의 차원 확인
    # [3, 32, 32] = [RGB Channel, Height, Width]

    # 전체 RGB 픽셀값 범위 확인
    print("Min:", images[i].min().item())
    print("Max:", images[i].max().item())

    # images[i] : i번째 이미지 Tensor
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
    # $$R_{mean} = \frac{1}{1024}\sum_{h=1}^{32}\sum_{w=1}^{32}R_{h,w}$$
    # 
    # RGB 평균값은 무엇을 의미하는가?
    # 예를 들어, R-0.65 / G-0.35 / B-0.20 이라 했을때 
    # 이 이미지에서는 평균적으로 R 채널의 픽셀값이 가장 높다는 뜻
    # 채널별 평균값을 비교하여 이미지의 전반적인 색상 분포 확인
    # 
    # 단, 평균값만으로 개별 픽셀의 색상이나 이미지 전체의 색상을 판단할 수는 없음
