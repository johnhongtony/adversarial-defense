
# ============================================================
# Day 4. 기존 VGG16 구조를 공부했던 경험을 바탕으로 간단한 CNN 직접 구현
# ============================================================
# 목표: CIFAR-10 이미지 분류에 사용할 CNN 모델의 구조를 이해하고 직접 구현한다.
#
# 1. CNN의 기본 구성 요소인 Conv2d, ReLU, MaxPool2d의 역할 이해
# 2. 각 Layer를 통과할 때 이미지의 크기와 채널 수가 어떻게 변하는지 계산
# 3. PyTorch의 nn.Module을 활용하여 SimpleCNN 모델 구현
# 4. 임의의 입력 데이터를 모델에 통과시켜 출력 형태와 구조 검증
# ============================================================


import torch.nn as nn

# CIFAR-10(3x32x32, 10클래스) 이미지 분류용 기본 CNN 모델
# PyTorch에서 신경망 모델을 만들 때 사용하는 기본 클래스
# SimpleCNN은 PyTorch의 nn.Module을 상속받은 모델이라는 뜻
class SimpleCNN(nn.Module):

    def __init__(self):
        # nn.Module 부모 클래스 초기화(파라미터/서브모듈 등록에 필수)
        super().__init__()

        # 1) 특징 추출기(Feature Extractor): 이미지에서 시각적 특징을 뽑는 구간
        self.features = nn.Sequential(
            # [입력: (B, 3, 32, 32)] -> [출력: (B, 32, 32, 32)]
            # 3채널(RGB)을 32개 필터로 스캔하며 3x3 지역 특징 추출, padding=1로 크기 유지
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # 음수 활성화 값을 0으로 잘라 비선형성 부여
            # [입력: (B, 32, 32, 32)] -> [출력: (B, 32, 16, 16)]
            # 2x2 구간마다 최댓값만 선택해 가로·세로 크기를 절반으로 축소
            nn.MaxPool2d(2),

            # [입력: (B, 32, 16, 16)] -> [출력: (B, 64, 16, 16)]
            # 채널을 32 -> 64로 늘려 더 복합적인 패턴 표현 학습
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # [입력: (B, 64, 16, 16)] -> [출력: (B, 64, 8, 8)]
            # 가로·세로 크기를 16 -> 8로 한 번 더 축소
            nn.MaxPool2d(2),

            # [입력: (B, 64, 8, 8)] -> [출력: (B, 128, 8, 8)]
            # 채널을 64 -> 128로 확장; 공간 해상도(8x8)는 유지
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 2) 분류기(Classifier): 추출된 특징을 바탕으로 10개 클래스 점수를 계산하는 구간
        self.classifier = nn.Sequential(
            # [입력: (B, 128, 8, 8)] -> [출력: (B, 128 * 8 * 8)] = (B, 8192)
            # 배치 차원(B)은 유지하고 나머지 (C, H, W)를 한 줄로 펼침
            nn.Flatten(),
            # [입력: (B, 8192)] -> [출력: (B, 256)]
            nn.Linear(in_features=128 * 8 * 8, out_features=256),
            nn.ReLU(),
            # [입력: (B, 256)] -> [출력: (B, 10)]
            # CIFAR-10 클래스 수(10개)에 맞춘 최종 출력층(소프트맥스 전 logits)
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        # 순전파 정의: 입력 텐서 x가 features를 거쳐 classifier까지 통과
        x = self.features(x)        # 공간 특징 맵 추출
        x = self.classifier(x)      # 10개 클래스 예측 점수 계산
        return x