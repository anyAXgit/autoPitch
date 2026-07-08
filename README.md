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

### 실촬영 다중 경기 (cam1/·cam2/ 폴더에 여러 경기)
`run_sessions.py`가 파일 촬영시각으로 캠을 경기별 페어링해 각각 렌더한다.
```bash
./.venv/bin/python run_sessions.py --list      # 페어링 미리보기
./.venv/bin/python run_sessions.py --game 1    # 1경기만
./.venv/bin/python run_sessions.py --all       # 전 경기 → data/output/game<N>/
```

## Tuning
`config.yaml`에서 임계값(`threshold_k`), 길이(`min_len_sec`/`max_len_sec`),
피크 상한(`max_clips`), 크로스페이드, 골 후 마진(`post_goal_sec`) 등 조정.

### 배경음악
`config.yaml`에 `bgm_path: data/bgm.mp3` 지정하면 `highlight_all.mp4`에 음악을
깔아준다(관중 소리는 그대로, 음악은 `bgm_volume`으로 낮게 믹싱, 길이에 맞춰 루프).

### 비전 골 확정 (선택)
`vision.enabled: true`면 골 후보 주변 프레임을 VLM에 보내 "골 가능성"을 판별해
오검출을 걸러낸다. `vision.model`(기본 opus, `claude-haiku-4-5`=저비용),
`vision.min_confidence`(예: 0.9로 정밀도 강화)로 조정. Anthropic API 키 필요.

### 네트-ROI 골 정밀 위치 (선택, 무API)
카메라가 고정이면 골망 위치도 고정 → 공이 네트를 때리는 **모션 스파이크**로
정확한 골 프레임을 잡는다. `editor/calibrate.html`을 열어 각 캠 정지 프레임에
골망 박스를 그려 `net_rois.json`을 만든 뒤 `locate.enabled: true` +
`locate.rois_path` 지정. 컷 앵커가 함성 onset → **진짜 골 프레임**으로 정밀화된다.
골망이 안 보이거나 스파이크가 약하면 **자동으로 onset 폴백**(클립은 유지).
정지 프레임 추출: `ffmpeg -ss 60 -i data/raw/cam1/DJI_...MP4 -frames:v 1 cam1.jpg`.

## Tests
```bash
./.venv/bin/python -m pytest -v
```
실제 영상 없이 합성 더미로 sync/peak/planner/render를 검증한다.

## plan.json & 편집 UI
편집 결정(컷 in/out·앵글)은 `plan.json`에 데이터로 저장된다.
`editor/index.html`을 브라우저로 열어 `plan.json`을 로드하면 클립별 포함/제외,
컷 지점(초) 수정, 순서 변경, (출력 폴더 지정 시) 미리보기를 할 수 있다.
"편집본 내보내기"로 나온 `plan.edited.json`을 다시 렌더:
```bash
./.venv/bin/python render_from_plan.py plan.edited.json data/output/edited [--bgm data/bgm.mp3]
```
