
# Python함수에 대한 공부  

## (visualize.py)Dataset과 DataLoader의 차이  

| 구분 | Dataset 직접 접근 | DataLoader |
|---|---|---|
| 접근 방식 | `train_dataset[i]` | `next(iter(train_loader))` |
| 데이터 개수 | 지정한 인덱스의 데이터 | Batch 단위 |
| 데이터 형태 | 이미지 1장 | 이미지 여러 장 |
| 주된 목적 | 개별 데이터 확인 | 모델 학습 및 평가 |  

예를 들어,  
 Batch Size가 64라면: 'images, labels = next(iter(train_loader))'
 이때 데이터 형태는 images: [64, 3, 32, 32], labels: [64]  

 반면 Dataset 직접 접근에서는 image: [3, 32, 32], label: 정수 1개  

* 핵심: Dataset은 개별 데이터를 제공하는 역할이고, DataLoader는 Dataset에서 데이터를 가져와 Batch 구성, 셔플, 병렬 로딩 등을 담당하는 역할.  

## (visualize.py)plt.subplots(n, n, figsize=(n, n))[ex.plt.subplots(4, 4, figsize=(10, 10))]  
 · Matplotlib에서 여러 개의 그래프를 하나의 화면에 배치하는 함수  
 · 사용 예: fig, axes = plt.subplots(4, 4, figsize=(10, 10))  
 <p align="center"> 
 <img src="./images/visualize/subplots.png" alt="matplotlib_4x4_subplot_structure" width="50%">
 </p>  




# Python함수 용어  

#### fig(figure): 그림 전체가 그려지는 커다란 흰색 도화지 한 장, 전체 그림을 관리하는 Figure 객체  
#### axes: 개별 이미지 출력 영역인 Axes 객체 배열(도화지 안에 세로 4줄 × 가로 4줄로 나눈 16개의 작은 액자(서브플롯) 모음)  
#### figsize: 전체 그림의 크기 (가로 10인치 × 세로 10인치), 크기를 안 키우면 그림과 제목이 서로 다닥다닥 붙어서 글자가 겹침  
####   
####   
####   
####   
####   
####   
####   
####   
####   
####   
####   
####   
