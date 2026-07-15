# autoPitch — 풋살 자동 하이라이트 파이프라인

> (구명: MAHP)

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
정확한 골 프레임을 잡는다. 캘리브레이션은 **각 카메라에 가까운 골망 하나만**
박스로(먼 골은 반대편 카메라가 맡음, 카메라당 1개) → `net_rois.json`. 이렇게 하면
각 골이 어느 캠 네트에서 났는지로 **골 난 쪽 카메라를 골 장면 앵글**로도 쓴다.
`locate.enabled: true` + `locate.rois_path` 지정 시 컷 앵커가 함성 onset →
**진짜 골 프레임**으로 정밀화된다. 골망이 안 보이거나 스파이크가 약하면
**자동으로 onset 폴백**(클립은 유지, Cam A 앵글).
정지 프레임 추출: `ffmpeg -ss 60 -i data/raw/cam1/DJI_...MP4 -frames:v 1 cam1.jpg`.

### ROI-only 조용한 골 스캔 (선택)
함성이 거의 없는 골을 찾기 위해 경기 전체의 네트 ROI 모션을 스캔한다
(`locate.scan_enabled: true`). 정크·비용을 3단계 캐스케이드로 억제한다:
1. **키프레임 러프 패스 + 결과 캐시** — 재분석 시 0초 (`locate.scan_cache`).
2. **임펄스 게이트(무료)** — 공-네트 히트는 짧은 임펄스(<`scan_max_impulse_sec`),
   키퍼가 네트를 만지는 지속 모션은 자동 컷.
3. **AI 판정(선택)** — `locate.scan_verify: vlm`이면 네트 클로즈업 3장을 VLM에
   "공이 네트에 들어왔나?"로 물어 최종 필터 (API 키 필요, 이벤트당 센트 미만).
   기본은 `shape`(2단계까지만) — 이때 후보는 편집기에 **ROI?** 배지로 표시되어
   영상으로 직접 확인·제외할 수 있다.
계획된 모든 이벤트는 `data/train_events.jsonl`에 자동 라벨(오디오 확인 여부)과
함께 쌓인다 — 데이터가 충분해지면 로컬 tiny 분류 모델 학습용.

## Tests
```bash
./.venv/bin/python -m pytest -v
```
실제 영상 없이 합성 더미로 sync/peak/planner/render를 검증한다.

## GUI 스튜디오 (권장)
루트 설정·골 ROI 캘리브레이션·후보 확인·영상 스크럽 미세편집·렌더를 한 곳에서.
```bash
./.venv/bin/python gui/server.py        # http://127.0.0.1:8756
```
로컬 서버(파이썬 stdlib, 추가 의존성 없음)가 ffmpeg/파이프라인을 돌리고 원본을
range 스트리밍하므로 브라우저에서 실제 영상을 스크럽하며 컷을 다듬을 수 있다.
탭: ① 설정(루트/경기) · ② 골 ROI(프레임에 박스 → net_rois.json) · ③ 편집·렌더.

## plan.json & 편집 UI (정적)
편집 결정(컷 in/out·앵글)은 `plan.json`에 데이터로 저장된다.
`editor/index.html`을 브라우저로 열어 `plan.json`을 로드하면 클립별 포함/제외,
컷 지점(초) 수정, 순서 변경, (출력 폴더 지정 시) 미리보기를 할 수 있다.
"편집본 내보내기"로 나온 `plan.edited.json`을 다시 렌더:
```bash
./.venv/bin/python render_from_plan.py plan.edited.json data/output/edited [--bgm data/bgm.mp3]
```
