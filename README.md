# MAHP — Matchwd Auto-Highlight Pipeline

동네 풋살 영상(단일캠 또는 3캠)에서 환호성 오디오 피크로 골을 찾아
자동으로 하이라이트 .mp4를 만드는 로컬 파이썬 파이프라인.

## Setup
```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Usage
1. `data/raw/`에 영상 넣기 (파일 1개 = 단일캠, 2+ = 멀티캠, 파일명 순 첫 번째가 기준 Cam A)
2. 실행:
   ```bash
   ./.venv/bin/python main.py
   ```
3. 결과: `data/output/highlight_<T>.mp4`, `highlight_all.mp4`, `plan.json`

## Tuning
`config.yaml`에서 임계값(`threshold_k`), 길이(`min_len_sec`/`max_len_sec`),
피크 상한(`max_clips`), 크로스페이드 등 조정.

## Tests
```bash
./.venv/bin/python -m pytest -v
```
실제 영상 없이 합성 더미로 sync/peak/planner/render를 검증한다.

## plan.json
편집 결정(컷 in/out·앵글)은 `plan.json`에 데이터로 저장된다. 향후 편집 UI(v2)가
이 파일을 로드해 컷을 조정한 뒤 `video_editor.render_plan`으로 재렌더링한다.
