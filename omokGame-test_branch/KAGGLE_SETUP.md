# Kaggle 실행 가이드 (Omok PPO)

## 1) 로컬에서 Kaggle 업로드용 zip 만들기

프로젝트 루트(`omokGame-test_branch`)에서 아래 중 하나로 압축:

- 파일 탐색기에서 폴더 우클릭 -> 압축(zip)
- 또는 터미널에서 zip 생성

압축 안에는 최소 아래가 포함되어야 합니다.

- `PPO_colab.py`
- `kaggle_run.py`
- `ai/`
- `core/`
- `main.py`

---

## 2) Kaggle Notebook 생성

1. Kaggle -> `Code` -> `New Notebook`
2. 우측 `Accelerator`를 `GPU`로 설정
3. 우측 `Add data` -> `Upload`에서 방금 zip 업로드

---

## 3) Notebook에서 코드 풀기 + 실행

첫 번째 셀:

```python
import os
import zipfile
from pathlib import Path

DATASET_DIR = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working/omok")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# /kaggle/input 에 있는 zip 자동 탐색
zip_files = []
for root, _, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith(".zip"):
            zip_files.append(Path(root) / f)

assert zip_files, "업로드한 zip 파일을 찾지 못했습니다."
zip_path = sorted(zip_files)[0]
print("Using zip:", zip_path)

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(WORK_DIR)

print("Extracted to:", WORK_DIR)
```

두 번째 셀:

```python
%cd /kaggle/working/omok
!python kaggle_run.py --resume
```

---

## 4) 체크포인트 이어서 학습하는 방법

핵심 파일:

- `checkpoints/ppo_p2.pt`
- `checkpoints/train_log.json`

학습이 끝난 뒤 `/kaggle/working`에도 복사됩니다.  
이 두 파일을 다운로드해서 **Kaggle Dataset으로 업로드**하면 다음 실행에서 이어서 학습할 수 있습니다.

---

## 5) 다음 실행(재개) 절차

1. 이전 실행에서 받은 `ppo_p2.pt`, `train_log.json`을 새 Dataset으로 업로드
2. Notebook 우측 `Add data`에서 그 Dataset 추가
3. 다시 `!python kaggle_run.py --resume` 실행

`kaggle_run.py`가 `/kaggle/input`에서 체크포인트를 찾아 자동 복원합니다.

