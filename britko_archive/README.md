# britko/WaferMap_Defect_Classification Archive

`britko/WaferMap_Defect_Classification` 저장소에서 가져온 파일들을 종류별로 정리한 폴더입니다.

## 폴더 구성

- `notebooks/`
  - 실험 기록, 버전별 모델 테스트, 이미지 분류 확인용 주피터 노트북 모음
- `python_scripts/`
  - 실제 실행 가능한 파이썬 스크립트 모음
- `docs/`
  - 참고 논문 또는 설명 문서 모음
- `config/`
  - 환경 설정이나 실행에 필요한 텍스트 파일 모음

## 파일별 설명

### notebooks
- `WaferMap_Defect_Classification.ipynb`: 웨이퍼맵 결함 분류 프로젝트 전체 흐름을 노트북 형태로 정리한 기본 실험 파일
- `load_model_WM.ipynb`: 저장된 웨이퍼맵 모델을 불러와 다시 사용하는 실험용 노트북
- `load_model_picture.ipynb`: 개별 이미지 파일을 넣어서 예측 결과를 확인하는 테스트용 노트북
- `project_v3(811k).ipynb`: WM-811K 데이터셋을 대상으로 진행한 3번째 버전 실험 노트북
- `project_v4(811k).ipynb`: WM-811K 데이터셋 기반의 4번째 버전 모델 실험 노트북
- `project_v5.ipynb`: 모델 구조나 학습 설정을 더 발전시킨 5번째 버전 실험 노트북
- `project_v5_new.ipynb`: `project_v5.ipynb`를 수정하거나 다시 정리한 개선판 노트북

### python_scripts
- `Project.py`: 프로젝트용 메인 파이썬 스크립트로, 데이터 처리나 모델 학습 흐름을 코드 형태로 실행하는 파일
- `load_model.py`: 저장된 학습 모델 파일을 불러와 예측이나 확인에 사용하는 스크립트
- `project_v2.py`: GitHub에서 핵심 학습 코드로 사용된 2번째 버전 CNN 학습 스크립트
- `tftf.py`: TensorFlow 동작 확인이나 간단한 테스트용으로 만든 보조 스크립트

### docs
- `Wafer_Map_Defect_Pattern_Classification_and_Image_Retrieval_Using_Convolutional_Neural_Network.pdf`: 웨이퍼 결함 분류와 이미지 검색 관련 내용을 설명하는 참고 논문 PDF

### config
- `Pipfile.txt`: 프로젝트 실행에 필요한 패키지나 환경 구성을 적어둔 설정 파일
