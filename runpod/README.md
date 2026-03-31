# RunPod Training Guide

이 폴더는 RunPod JupyterLab에서 `project_v2.py` 스타일 CNN을 바로 학습시키기 위한 파일들입니다.

## 준비된 파일

- `train_project_v2_halves.py`
  - `datasets/LSWMD_paired_split_7_1_2_half_1` 또는 `half_2`를 받아 학습, 검증, 테스트 평가까지 한 번에 수행하는 스크립트

## RunPod Jupyter에서 실행 순서

### 1. 저장소 받기

```bash
%cd /workspace
!git clone https://github.com/shingidong/semiconductor_waper_detect.git
%cd /workspace/semiconductor_waper_detect
```

### 2. 패키지 설치

```bash
!pip install -U pip
!pip install tensorflow scikit-learn matplotlib
```

### 3. GPU 확인

```bash
!nvidia-smi
```

### 4. `half_1` 학습

```bash
!python runpod/train_project_v2_halves.py \
  --data-root /workspace/semiconductor_waper_detect/datasets/LSWMD_paired_split_7_1_2_half_1 \
  --output-dir /workspace/outputs/half_1 \
  --epochs 10 \
  --batch-size 32 \
  --initial-weights /workspace/semiconductor_waper_detect/checkpoint_v2-07-0.55-0.82.h5
```

### 5. `half_2` 학습

```bash
!python runpod/train_project_v2_halves.py \
  --data-root /workspace/semiconductor_waper_detect/datasets/LSWMD_paired_split_7_1_2_half_2 \
  --output-dir /workspace/outputs/half_2 \
  --epochs 10 \
  --batch-size 32 \
  --initial-weights /workspace/semiconductor_waper_detect/checkpoint_v2-07-0.55-0.82.h5
```

## 결과물

각 출력 폴더에는 아래 파일이 저장됩니다.

- `best_model.keras`
  - 검증 손실이 가장 좋았던 최종 모델
- `training_log.csv`
  - epoch별 loss, accuracy 로그
- `training_curves.png`
  - 학습/검증 loss, accuracy 그래프
- `classification_report.txt`
  - test 데이터 기준 precision, recall, f1-score 요약
- `confusion_matrix.png`
  - 테스트 결과 confusion matrix 시각화
- `metrics.json`
  - test loss, accuracy, 클래스 인덱스, confusion matrix를 저장한 JSON 파일

## 참고

- 이미지 크기는 기본 `64x64`입니다.
- 배치 크기는 기본 `32`입니다.
- `checkpoint_v2-07-0.55-0.82.h5`를 초기 가중치로 불러오게 해두었지만, 원하면 `--initial-weights`를 빼고 처음부터 학습할 수도 있습니다.
