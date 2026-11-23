

# ============================================================
# Day 2. CIFAR-10 데이터셋 이해 및 로딩
# ============================================================
# 목표:
# 1. 딥러닝 모델을 학습시키기 전에 이미지 데이터를 불러온다
# 2. 불러온 이미지를 전처리
# 3. 전처리된 이미지를 일정한 크기의 묶음(Batch)으로 공급하는 데이터 파이프라인(Data Pipeline)을 구축
# ============================================================



# 라이브러리 import
import torch
# torch: Tensor 연산, GPU 연산, 딥러닝 모델 구현
from torchvision import datasets, transforms
# datasets:	CIFAR-10 등 데이터셋 불러오기, transforms: 이미지 전처리
from torch.utils.data import DataLoader
# DataLoader: 데이터를 Batch 단위로 공급
# 즉, Dataset: 데이터 자체를 관리, Transform: 데이터를 모델에 적합한 형태로 변환, DataLoader: 데이터를 Batch 단위로 제공

# ② 이미지 전처리 
transform = transforms.ToTensor()
# ToTensor()는 이미지 데이터를 PyTorch Tensor로 변환하는 함수
# 주요 작업 세 가지
# 1. 이미지 데이터를 Tensor로 변환
# 2. 이미지 차원을 HWC에서 CHW로 변경
# 3. 픽셀값을 0~255에서 0~1 범위로 변환

# ③ 학습 데이터 불러오기
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
# 각 인자의 의미는 다음과 같음
# root="./data" : 데이터 저장 위치
# train=True: 학습 데이터 선택
# download=True: 데이터가 없으면 다운로드
# transform=transform: 앞에서 정의한 이미지 전처리 적용

# ④ 테스트 데이터 불러오기
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
# 학습 데이터와 거의 동일하나, 차이점은 train=False 즉, 학습에 사용하지 않은 별도 이미지로 모델의 일반화 성능 평가

# ⑤ DataLoader 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
# Batch란 50,000장의 이미지를 한꺼번에 모델에 넣는 대신 일정한 크기로 나누어 처리하는 단위
# batch_size=64 로 설정하여 이미지 64장을 하나의 배치로 묶음(50,000장 - 1배치 64장, 총 782개 배치 중 마자막배치는 32장)
# shuffle=True 를 주어, 매 Epoch마다 데이터 순서를 섞어 모델이 특정 순서에 지나치게 의존하지 않도록 함

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
# 테스트에서는 데이터 순서를 섞지 않아도 평가에 문제가 없으므로 shuffle=False로 설정

# ⑥ DataLoader에서 실제 데이터 가져오기
images, labels = next(iter(train_loader))
# 순서대로 살펴보면 다음과 같음
# 1. iter(train_loader): DataLoader에서 데이터를 하나씩 꺼낼 수 있도록 반복자(Iterator)를 생성
# 2. next(...): 반복자에서 다음 데이터를 가져옴. 여기서는 첫 번째 Batch를 가져옴
# 3. images, labels: 가져온 Batch를 이미지와 정답 레이블로 분리, 따라서 결과적으로 다음 두 가지 데이터가 생성 images: 이미지 64장, labels: 각 이미지의 정답 클래스 64개

# ⑦ 이미지 Tensor의 구조
print("Image Shape:", images.shape)
# 이전에 공부했던 VGG16의 입력 데이터 구조와 연결됨(예상출력 [64, 3, 32, 32)
# 64	Batch Size / 3	RGB 채널 / 32	Height / 32 	Width
# 즉, 이미지 64장을 각각 3채널, 32×32 크기의 Tensor로 표현

# ⑧ Label의 구조
print("Label Shape:", labels.shape)
# 예상 출력: [64]
# 이미지 64장에 대응하는 정답이 64개라는 뜻

# ⑨ 이미지 최솟값과 최댓값
print("Image Min:", images.min())
print("Image Max:", images.max())
# 이미지 Tensor에 포함된 모든 픽셀값 중 최솟값과 최댓값을 확인하는 코드
# ToTensor()를 적용했으므로 값은 0~1 범위에 있어야 함

print("Train Dataset Size:", len(train_dataset))
print("Test Dataset Size:", len(test_dataset))

print("Images Device:", images.device)
print("Images Dtype:", images.dtype)

print("First Label:", labels[0].item())
print("First Image Shape:", images[0].shape)