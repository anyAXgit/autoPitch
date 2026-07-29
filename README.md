# autoPitch — 풋살 자동 하이라이트 파이프라인

> (구명: MAHP)

동네 풋살 영상(단일캠부터 4캠까지)에서 환호성 오디오 피크와 골망 ROI로 골을 찾아
자동으로 하이라이트 .mp4를 만드는 로컬 파이썬 파이프라인.

## Setup
```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Usage
1. `data/raw/`에 영상 넣기
   - 날짜별 권장 구조:
     ```text
     data/raw/
       2026-07-03/
         cam1/*.MP4
         cam2/*.MOV
       2026-07-10/
         cam1/*.MP4
         cam2/*.MOV
     ```
   - 기존 flat 구조도 지원:
     ```text
     data/raw/
       cam1/*.MP4
       cam2/*.MOV
     ```
   - `cam1`부터 `cam4`까지 지원. 단일캠은 `cam1`만 두면 된다.
2. 실행:
   ```bash
   ./.venv/bin/python main.py
   ```
3. 결과: `data/output/highlight_<T>.mp4`, `highlight_all.mp4`, `plan.json`

### 실촬영 다중 경기
여러 경기 파일이 한 날짜 폴더에 들어 있으면 파일 촬영시각으로 같은 경기의
카메라들을 페어링한다. 날짜별 폴더가 있으면 날짜마다 경기 1, 2, 3...으로
분리되고, 기존 `data/raw/cam1`, `data/raw/cam2` 구조는 파일 메타데이터에서
날짜 라벨을 추론해 하나의 날짜 세션으로 표시한다.

`run_sessions.py`는 같은 페어링 규칙으로 CLI 렌더를 수행한다.
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
탭: ① 설정(루트/날짜별 경기) · ② 골 ROI(프레임에 박스 → net_rois.json) · ③ 편집·렌더.

GUI의 편집 탭은 `날짜` 선택 후 해당 날짜의 `경기`만 보여준다. 내부 분석 캐시는
`날짜세션:경기번호` 기준으로 분리되므로 서로 다른 날짜의 경기 1이 충돌하지 않는다.
날짜 폴더가 없는 기존 구조도 그대로 열리며, 이 경우 촬영시각 메타데이터에서
예: `2026-07-03` 같은 날짜 라벨을 자동 추론한다.

ROI 탭의 `좌우 반전`은 파일별 보기 설정이다. 같은 날짜의 같은 cam 전체에 적용할
수 있고, 설정은 `data/_gui/view_settings.json`에 저장된다. ROI 화면은 반전된 보기로
박스를 잡되 좌표는 원본 기준으로 저장하며, 편집 미리보기와 최종 렌더에도 같은
좌우 반전이 적용된다.

## plan.json & 편집 UI (정적)
편집 결정(컷 in/out·앵글)은 `plan.json`에 데이터로 저장된다.
`editor/index.html`을 브라우저로 열어 `plan.json`을 로드하면 클립별 포함/제외,
컷 지점(초) 수정, 순서 변경, (출력 폴더 지정 시) 미리보기를 할 수 있다.
"편집본 내보내기"로 나온 `plan.edited.json`을 다시 렌더:
```bash
./.venv/bin/python render_from_plan.py plan.edited.json data/output/edited [--bgm data/bgm.mp3]
```

## 라이선스
[MIT](LICENSE) © anyax

의존성 참고:
- 핵심 파이프라인 의존성(numpy·scipy·librosa·soundfile·PyYAML·Pillow·anthropic)은 모두 허용형(MIT/BSD류).
- **ffmpeg**은 시스템에 별도 설치해 서브프로세스로 호출한다(코드에 링크·번들하지 않으므로 라이선스가 전이되지 않음).
- **`tools/kickoff_scan.py`** 는 선택적 실험 프로토타입으로 `ultralytics`(YOLO, **AGPL-3.0**)를 쓴다. 코어 파이프라인은 이를 사용하지 않으며 사용자가 직접 설치해야 한다. 이 도구를 쓰거나 배포하면 해당 부분에 한해 AGPL-3.0 의무가 따른다.

---

## 설치판 (macOS / Windows)

배포된 앱을 쓰는 경우 파이썬 설치가 필요 없다. 실행하면 홈 폴더에
`~/autoPitch` 작업 공간을 만들고 브라우저가 열린다.

첫 화면이 환경을 점검한다 — ffmpeg 유무, 필요한 필터, 하드웨어 인코더,
폴더 권한, 디스크 여유. 문제가 있으면 해결 방법을 그 자리에 보여주고,
`설치했습니다 · 다시 확인` 버튼으로 재점검한다.

### 직접 빌드

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python packaging/build.py --clean
```

결과는 `dist/`에 생긴다 (macOS `autoPitch.app`, Windows `autoPitch.exe`).
PyInstaller는 크로스 컴파일을 못 하므로 각 OS에서 따로 빌드해야 한다.

**ffmpeg 포함 여부** — 기본 빌드는 ffmpeg를 넣지 않고 PATH에서 찾는다.
없으면 첫 화면이 설치 방법(`brew install ffmpeg` / `winget install --id
Gyan.FFmpeg`)을 안내한다. 자체 포함하려면:

```bash
./.venv/bin/python packaging/build.py --vendor-ffmpeg $(which ffmpeg) $(which ffprobe)
```

libx264가 포함된 ffmpeg 빌드는 GPL이다. 재배포하면 해당 바이너리에 대한
GPL 의무(라이선스 전문·소스 제공)를 지게 된다. 이 프로젝트 코드는 별도
프로세스로 호출하므로 MIT를 유지한다.
