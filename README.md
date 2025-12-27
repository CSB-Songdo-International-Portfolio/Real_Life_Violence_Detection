# TensorFlow와 YOLO를 활용한 CCTV 분석 스마트보안시스템

> **AI 기반 실시간 폭력 범죄 감지 및 자동 신고 시스템**

## 📋 개요

본 프로젝트는 **딥러닝 기술**을 활용하여 CCTV 영상을 실시간으로 분석하고, 폭력 범죄를 자동으로 감지하여 신고하는 **스마트 보안 시스템**입니다.

### 🎯 연구 배경

- **현황**: 2021년 대한민국에서 총 1,429,826건의 범죄 발생 (검거율 79.5%)
- **문제점**: 약 1,600만 대의 CCTV가 사후 확인용으로만 활용 중
- **솔루션**: AI를 활용한 **사전적 범죄 예방** 및 **자동 신고 시스템** 구축

---

## 🔧 기술 스택

| 분류 | 기술 |
|------|------|
| **프레임워크** | TensorFlow, Keras, PyTorch |
| **딥러닝 모델** | MobileNet + Bidirectional LSTM, YOLOv8 |
| **라이브러리** | OpenCV, NumPy, scikit-learn, Seaborn |
| **개발 환경** | Python, Jupyter Notebook |
| **GPU** | NVIDIA (MPS 지원) |

---

## 📊 시스템 아키텍처

### 1️⃣ 폭력 상황 분류 모델 (Research 1)

**목표**: 비디오 프레임에서 폭력/비폭력 상황 자동 분류

#### 데이터셋
- **출처**: Kaggle - "Real Life Violence Situations Dataset"
- **구성**: 비폭력 1,000개 + 폭력 1,000개 영상
- **전처리**: 
  - 해상도: 64×64
  - 프레임 추출: 영상당 16개 이미지
  - 정규화: 픽셀값 0~1 범위

#### 모델 구조

```
Input (16 프레임, 64×64×3)
    ↓
TimeDistributed(MobileNet)
    ↓
Bidirectional LSTM (forward: 32, backward: 32)
    ↓
Dense 층 (256 → 128 → 64 → 32)
    ↓
Output (Softmax, 2-class 분류)
```

#### 성능 지표
- **훈련 정확도**: 94.03%
- **검증 정확도**: 91.11%
- **테스트 정확도**: 90.00%
- **테스트 손실**: 0.2982

---

### 2️⃣ 물체 인식 모델 (Research 2)

**목표**: 흉기(칼, 각목, 쇠파이프 등) 자동 감지

#### YOLOv8 설정
```python
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
model = YOLO("yolov8n.pt")
model.train(data, epochs=100, imgsz=640, device="mps")
```

#### 감지 대상
- 칼
- 각목
- 쇠파이프
- 기타 흉기

---

### 3️⃣ 자동 신고 메커니즘

```
CCTV 영상 입력
    ↓
폭력 상황 감지 (MobileNet + LSTM)
    ↓
흉기 인식 (YOLO)
    ↓
영상 판정 전담 부서 (2차 검증)
    ↓
경찰청 종합상황실 신고
    ↓
현장 경찰 출동 및 초동조치
```

---

## 🚀 주요 기능

### ✅ 1. 프레임 추출 및 전처리

```python
def frames_extraction(video_path):
    """비디오에서 프레임 추출 및 정규화"""
    - cv2.VideoCapture로 비디오 읽기
    - 전체 프레임 수 계산
    - skip_frames_window로 균등 간격 추출
    - 픽셀값 정규화 (0~1)
    return normalized_frame
```

### ✅ 2. 데이터셋 생성

```python
def create_dataset():
    """클래스별 비디오 처리 및 프레임 데이터 생성"""
    - 각 클래스 폴더 순회
    - frames_extraction 함수 호출
    - features, labels, 경로 저장
    - NumPy 배열로 변환 후 .npy 저장
```

### ✅ 3. 모델 학습

```python
def MobileNet_Add_LSTM_model():
    """MobileNet + Bidirectional LSTM 모델"""
    - MobileNet: 이미지 특징 추출
    - TimeDistributed: 각 프레임 처리
    - Bidirectional LSTM: 시간적 특성 학습
    - Dense 레이어: 분류
```

### ✅ 4. 실시간 비디오 예측

```python
def predict_frames(video_file_path, output_path):
    """실시간 프레임 분석 및 시각화"""
    - 프레임 읽기 및 정규화
    - 큐(deque)에 저장
    - 모델 예측
    - 결과 시각화 및 저장
```

### ✅ 5. 성능 평가

```python
def model_predict_history():
    """학습 곡선 및 혼동 행렬 시각화"""
    - Loss vs Validation Loss
    - Accuracy vs Validation Accuracy
    - Confusion Matrix 분석
```

---

## 📈 실험 결과

### 폭력 상황 분류 결과

| 메트릭 | 훈련 | 검증 | 테스트 |
|--------|------|------|--------|
| **정확도** | 94.03% | 91.11% | 90.00% |
| **손실** | 0.1718 | 0.2974 | 0.2982 |

### 혼동 행렬 (Confusion Matrix)
- 비폭력 상황: 높은 정확도로 올바르게 분류
- 폭력 상황: 일부 오분류 존재하나 전반적 우수한 성능

---

## 💾 설치 및 실행

### 필수 요구사항

```bash
# 패키지 설치
pip install tensorflow keras opencv-python numpy scikit-learn seaborn
pip install torch torchvision ultralytics

# GPU 지원 (선택사항)
pip install tensorflow-gpu  # CUDA 필수
```

### 데이터셋 준비

```bash
# Kaggle에서 "Real Life Violence Dataset" 다운로드
# 폴더 구조:
# dataset/
#   ├── Violence/
#   │   ├── V_1.mp4
#   │   └── ...
#   └── NonViolence/
#       ├── NV_1.mp4
#       └── ...
```

### 모델 학습

```python
# 1. 데이터셋 생성
features, labels, paths = create_dataset(data_path)

# 2. 모델 구성 및 학습
model = MobileNet_Add_LSTM_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(features_train, labels_train, epochs=100, callbacks=[early_stopping, reduce_lr])

# 3. 모델 평가
model.evaluate(features_test, labels_test)

# 4. 비디오 예측
predict_frames(input_video_path, output_video_path, image_cnt=16)
```

---

## 🔍 주요 선행 연구

1. **딥러닝 기반 범죄자 신원 인식 시스템** (조나래, 2017)
   - 얼굴 영역 추출 및 특징 벡터 추출 방법론 참고

2. **딥러닝 기반 CCTV 이상행동 감지 시스템** (이용주, 2022)
   - 3D ResNet + UCF-Crime 데이터셋
   - 폭력적 동작 감지 기법 활용

---

## 🎓 이론적 배경

### CCTV (폐쇄회로 텔레비전)
- 특정 목적의 폐쇄망 영상 시스템
- 카메라 + DVR(영상 녹화 장비) 구성
- 기존 용도: 사건 확인(사후), 감시, 화재 예방

### YOLO (You Only Look Once)
**특징**:
- 이미지 전체를 한 번만 분석
- 4단계 프로세스 통합: 영역 제안 → 특징 추출 → 분류 → 바운딩박스 회귀
- **장점**: 매우 빠른 처리 속도, 높은 정확도

**작동 원리**:
1. 입력 이미지를 S×S 그리드로 분할
2. 각 그리드에서 물체 위치(Bounding Box) 및 신뢰도(Confidence Score) 예측
3. 높은 신뢰도의 박스 시각화
4. NMS(Non-Maximum Suppression)로 최적 결과 도출

---

## 💡 시스템 개선 방안

### 단기 개선안
- ✅ 더 많은 훈련 데이터 수집 및 정제
- ✅ 다양한 흉기 데이터셋 확보
- ✅ 다양한 테스트 데이터셋에서 일반화 능력 검증
- ✅ 속도, 실시간 처리 능력 최적화

### 장기 비전
- 🔮 웹 플랫폼 연계
- 🔮 지역별 실시간 영상 제공
- 🔮 공공 안전 데이터 통합
- 🔮 모바일 애플리케이션 개발

---

## 📝 결론

본 연구를 통해 **AI 기반 CCTV 분석이 실질적으로 가능**함을 증명했습니다:

1. **90% 이상의 높은 정확도** 달성
2. **YOLO를 통한 효과적인 흉기 인식**
3. **인간의 판정 오류를 보완**하는 2차 검증 시스템 제안
4. **기존 사후 대응 방식**에서 **사전적 예방**으로의 패러다임 전환

---

## 📂 파일 구조

```
.
├── README.md
├── data/
│   ├── Violence/           # 폭력 상황 비디오
│   └── NonViolence/        # 정상 상황 비디오
├── models/
│   ├── mobilenet_lstm.h5   # 학습된 폭력 분류 모델
│   └── yolov8n.pt          # YOLO 물체 감지 모델
├── src/
│   ├── preprocessing.py     # 프레임 추출 및 전처리
│   ├── model.py            # 모델 정의 및 학습
│   ├── inference.py        # 예측 및 시각화
│   └── config.py           # 설정값
└── notebooks/
    └── analysis.ipynb      # Jupyter 분석 노트북
```

---

## 🔗 참고자료

- [1] Kaggle: Real-time Violence Detection - MobileNet + Bi-LSTM
- [2] 조나래 (2017). 딥러닝 기반 범죄자 신원 인식 시스템 설계 및 구현. 가천대학교
- [3] 이용주 (2022). 딥러닝 기반 CCTV 이상행동 감지 시스템 설계. 한국지식정보기술학회
- [4] Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics

---
