# autoPitch

두 시간짜리 풋살 영상에서 하이라이트 후보를 뽑아 컷으로 만들어 줍니다.
다 돌려볼 필요 없이, 나온 후보를 확인해서 넣고 빼면 됩니다.

소리로 찾습니다. 골이 들어가면 사람들이 반응하고 그 순간 오디오가 튑니다 —
화면을 볼 필요가 없습니다. 여기에 고정 카메라라는 공짜 정보를 얹어(골대 위치가
늘 같으니) 컷 시작점을 프레임 단위로 맞춥니다.

- **로컬에서 돕니다.** 촬영본이 어디로도 올라가지 않습니다.
- **거창한 장비가 필요 없습니다.** 액션캠·휴대폰 한두 대를 펜스에 물리면 됩니다.
- **골만 모으지 않습니다.** 선방, 아깝게 빗나간 슛 — 시끄러웠던 순간이 다 후보로
  올라오고 편집기에서 넣고 뺍니다.

## 어디까지 되는가

실측한 것만 적습니다.

| | |
|---|---|
| 두 카메라 시각 맞추기 | 현장음 상호상관으로 자동. 메타데이터 없이 맞춥니다 |
| 반응 지연 보정 | 두 구장 140골 이상에서 소리는 골보다 중앙값 1.5초 늦게 터집니다. 그만큼 앞으로 당겨 자릅니다 |
| 다른 구장 | 골대 위치만 다시 그리면 그대로 동작합니다 |
| 골이 아닌 장면 | 버그가 아닙니다. 선방·아쉬운 슛도 후보로 올리고 편집기에서 넣고 뺍니다 |

소리가 기준이라 **아무도 반응하지 않은 장면은 찾지 못합니다.** 카메라를 골대
뒤에 하나 더 두면 달라질 수 있고 그건 아직 해보지 않았습니다.

## 필요한 것

- **ffmpeg** — 없으면 첫 화면이 알려주고 버튼 하나로 설치합니다.
- **Python 3.10 이상** — 소스로 돌릴 때만. 설치판은 파이썬이 없어도 됩니다.
- (선택) **Anthropic API 키** — 골 라벨링을 켤 때만.

## 설치

**설치판** — [Releases](https://github.com/anyAXgit/autoPitch/releases)에서
macOS / Windows 빌드를 받습니다. 실행하면 홈에 `~/autoPitch` 작업 공간이 생기고
브라우저가 열립니다. 첫 화면이 환경을 점검합니다.

> **아직 코드 서명 전이라 첫 실행에 한 번 막힙니다.**
> macOS: 실행이 차단되면 시스템 설정 → 개인정보 보호 및 보안 을 열고 아래로
> 내려가 `확인 없이 열기`. (macOS 15부터는 우클릭 → 열기 로 넘어가지 않습니다.)
> Windows: SmartScreen 창에서 `추가 정보` → `실행`.

**소스에서**

```bash
git clone https://github.com/anyAXgit/autoPitch.git
cd autoPitch
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python autopitch.py          # 브라우저 스튜디오
```

ffmpeg 가 따로 필요합니다 (`brew install ffmpeg` / `winget install --id
Gyan.FFmpeg` / `apt install ffmpeg`).

## 쓰는 법

1. **카메라별로 촬영본이 든 폴더를 지정합니다.** 복사하지 않고 그 자리에서
   읽습니다. 지정하지 않으면 작업 폴더의 `data/raw/<날짜>/cam1`, `cam2` … 를
   씁니다 (단일캠이면 `cam1` 만).
2. 골대 위치를 한 번 그립니다. 건너뛰어도 하이라이트는 나옵니다.
3. `장면 찾기` → 나온 후보를 확인·수정 → 내보내기.

카메라는 **대각선으로 마주 보는 코너**에 두고 경기 중에는 움직이지 않게 고정합니다.
설정 화면의 *카메라 어떻게 놓을까* 에 그림과 함께 정리해 두었습니다.

명령줄로도 됩니다:

```bash
./.venv/bin/python main.py                # data/raw 전체
./.venv/bin/python run_sessions.py --all  # 날짜별 여러 경기
```

## 더 자세히

### 설정 조정
`config.yaml`에서 임계값(`threshold_k`), 길이(`min_len_sec`/`max_len_sec`),
피크 상한(`max_clips`), 크로스페이드, 골 후 마진(`post_goal_sec`) 등 조정.
자주 만지는 값은 GUI 의 `세부 설정`에도 있다. 거기서 저장해도 이 파일에 적어 둔
주석과 정렬은 그대로 남는다 — 왜 그 값인지가 주석에 적혀 있어서다.

### 배경음악
`config.yaml`에 `bgm_path: data/bgm.mp3` 지정하면 `highlight_all.mp4`에 음악을
깔아준다(관중 소리는 그대로, 음악은 `bgm_volume`으로 낮게 믹싱, 길이에 맞춰 루프).

### 골 라벨링 (선택)
`vision.enabled: true` 면 후보 주변 프레임 몇 장을 VLM 에 보내 골인지 물어보고,
그 답을 `vision_goal` · `vision_conf` 로 **기록만 한다.** 클립을 지우지 않고
화면에 배지를 그리지도 않는다 — 소리로 잡은 후보가 골이 아닐 수 있는 것처럼
모델의 판정도 틀릴 수 있어서, 최종 판단은 편집기에서 사람이 한다. 정렬·검토와
나중의 학습 데이터로 쓰는 값이다.

`vision.model` 기본값은 `claude-sonnet-5`(`claude-haiku-4-5` 로 바꾸면 싸진다),
`vision.min_confidence` 를 올리면(예: 0.9) 그 아래 확신도는 골 아님으로 적는다.

키와 패키지가 따로 필요하다:

```bash
./.venv/bin/pip install anthropic
export ANTHROPIC_API_KEY=...     # 또는 작업 폴더의 data/_gui/anthropic_key.txt
```

### 네트-ROI 골 정밀 위치 (선택, 무API)
카메라가 고정이면 골망 위치도 고정 → 공이 네트를 때리는 **모션 스파이크**로
정확한 골 프레임을 잡는다. 캘리브레이션은 **각 카메라에 가까운 골망 하나만**
박스로(먼 골은 반대편 카메라가 맡음, 카메라당 1개) → `net_rois.json`. 이렇게 하면
각 골이 어느 캠 네트에서 났는지로 **골 난 쪽 카메라를 골 장면 앵글**로도 쓴다.
`locate.enabled: true` + `locate.rois_path` 지정 시 컷 앵커가 함성 onset →
**진짜 골 프레임**으로 정밀해진다. 골망이 안 보이거나 스파이크가 약하면
**자동으로 onset 폴백**(클립은 유지, Cam A 앵글).
정지 프레임 추출: `ffmpeg -ss 60 -i data/raw/cam1/DJI_...MP4 -frames:v 1 cam1.jpg`.

### 조용한 골 스캔 (기본 꺼짐)
반응이 없는 골을 골대 영역의 움직임만으로 찾아보려는 실험이다.
`locate.scan_enabled`로 켤 수 있고 **기본값은 꺼짐**이다 — 개발에 쓴 카메라
배치(측면 낮은 코너)에서는 골과 잡음이 갈리지 않았다. 이유와 시도한 방법은
`config.yaml`의 해당 항목에 주석으로 적어 두었다.

켜면 키프레임 러프 패스 → 임펄스 게이트 → (선택) VLM 판정의 3단계로 후보를
줄이고 남은 것은 편집기에 **ROI?** 배지로 떠서 직접 확인·제외할 수 있다.
카메라를 골대 뒤에 두는 배치라면 다시 볼 만하다.

판정 결과는 `data/train_events.jsonl`에 쌓인다 — 충분히 모이면 그 구장에 맞는
가벼운 분류 모델을 학습시킬 수 있다.

### GUI 스튜디오
루트 설정·골 ROI 캘리브레이션·후보 확인·영상 스크럽 미세편집·렌더를 한 곳에서.
```bash
./.venv/bin/python gui/server.py        # http://127.0.0.1:8756
```
로컬 서버(파이썬 stdlib, 추가 의존성 없음)가 ffmpeg/파이프라인을 돌리고 원본을
range 스트리밍하므로 브라우저에서 실제 영상을 스크럽하며 컷을 다듬을 수 있다.
탭: ① 설정(루트/날짜별 경기) · ② 골 ROI(프레임에 박스 → net_rois.json) · ③ 편집·렌더.

서버는 `127.0.0.1`에만 붙고, **자기 페이지에서 온 요청만** 처리한다
(`Host`·`Origin`·`Sec-Fetch-Site` 확인). 루프백 바인드만으로는 다른 기계를 막을
뿐 브라우저의 다른 탭을 못 막기 때문이다.

GUI의 편집 탭은 `날짜` 선택 후 해당 날짜의 `경기`만 보여준다. 내부 분석 캐시는
`날짜세션:경기번호` 기준으로 분리되므로 서로 다른 날짜의 경기 1이 충돌하지 않는다.
날짜 폴더가 없는 기존 구조도 그대로 열리며 이 경우 촬영시각 메타데이터에서
`2026-07-03` 같은 날짜 라벨을 자동 추론한다.

ROI 탭의 `좌우 반전`은 파일별 보기 설정이다. 같은 날짜의 같은 cam 전체에 적용할
수 있고 설정은 `data/_gui/view_settings.json`에 저장된다. ROI 화면은 반전된 보기로
박스를 잡되 좌표는 원본 기준으로 저장하며 편집 미리보기와 최종 렌더에도 같은
좌우 반전이 적용된다.

### plan.json · 정적 편집 UI
편집 결정(컷 in/out·앵글)은 `plan.json`에 데이터로 저장된다.
`editor/index.html`을 브라우저로 열어 `plan.json`을 로드하면 클립별 포함/제외,
컷 지점(초) 수정, 순서 변경, (출력 폴더 지정 시) 미리보기가 된다.
"편집본 내보내기"로 나온 `plan.edited.json`을 다시 렌더:
```bash
./.venv/bin/python render_from_plan.py plan.edited.json data/output/edited [--bgm data/bgm.mp3]
```

### 테스트
```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest -q
```
실제 영상 없이 합성 더미로 sync/peak/planner/render를 검증한다.

## 기여

버그 신고와 다른 구장에서의 결과 보고를 환영합니다.
먼저 [CONTRIBUTING.md](CONTRIBUTING.md) 를 봐 주세요.
이슈 템플릿에 **"다른 구장에서 돌려봤습니다"** 가 따로 있습니다 — 어떤 배치에서
되고 안 되는지가 이 프로젝트에 제일 필요한 데이터입니다.

변경 내역은 [CHANGELOG.md](CHANGELOG.md).

## 라이선스

[MIT](LICENSE) © anyax

- 코어는 numpy · scipy · Pillow · PyYAML 뿐이고 전부 허용형(BSD/MIT)이다.
- **ffmpeg** 은 번들하지 않고 별도 프로세스로 호출한다 — 라이선스가 전이되지
  않는다. 왜 번들하지 않기로 했는지는 아래에 적어 두었다.
- **`src/player_analysis.py`**, **`tools/kickoff_scan.py`** 는 선택·실험 모듈로
  `ultralytics`(**AGPL-3.0**)를 쓴다. 코어와 배포 앱은 이를 임포트하지 않고
  `requirements.txt` 에도 없다. 직접 설치해 쓰는 경우 그 의무는 사용자에게 있다.

---

## 직접 빌드하기

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python packaging/build.py --clean
```

결과는 `dist/`에 생긴다 (macOS `autoPitch.app`, Windows `autoPitch.exe`).
PyInstaller는 크로스 컴파일을 못 하므로 각 OS에서 따로 빌드해야 한다.

### ffmpeg를 왜 번들하지 않는가

**결정: 배포 빌드에 ffmpeg를 넣지 않는다.** 첫 화면이 유무를 확인하고
없으면 `ffmpeg 설치하기` 버튼 한 번으로 각 OS의 패키지 관리자로 설치한다.

측정해 보면 파이프라인이 요구하는 것 중 **GPL인 건 `libx264` 하나뿐**이다.
aac 인코딩, xfade/acrossfade, overlay/scale/pad, 하드웨어 인코더
(videotoolbox·nvenc·qsv·amf)는 전부 LGPL 범위다. 실제로 하드웨어 인코더만으로
전체 렌더가 완결되는 것을 확인했다.

그래서 세 갈래가 있었다:

| | HW 인코더 있는 기기 | HW 인코더 없는 기기 | 라이선스 부담 |
|---|---|---|---|
| LGPL 번들 (x264 제외) | 동작 | **렌더 불가** | LGPL 고지·소스 제공 |
| GPL 번들 (x264 포함) | 동작 | 동작 | **GPL 소스 제공 의무** |
| 번들 안 함 (채택) | 동작 | 동작 | 없음 |

LGPL 번들은 "느려짐"이 아니라 **아예 못 쓰는 기기**를 만든다(GitHub의 Windows
러너가 정확히 그런 환경이다). openh264를 넣으면 해결되지만 macOS용은 직접
빌드해야 하고 특허 관계도 단순하지 않다. GPL 번들은 동작하지만 MIT 프로젝트의
릴리스에 GPL 바이너리를 넣고 그 소스 제공 의무를 지는 일이다.

한편 앱은 아직 양쪽 OS에서 **서명되어 있지 않다**(adhoc). 어느 쪽이든 첫 실행에
경고를 한 번 거치므로, 번들해도 "더블클릭하면 그냥 됨"은 성립하지 않는다.
그 대가로 GPL을 질 이유가 없다고 판단했다.

직접 번들하려면:

```bash
./.venv/bin/python packaging/build.py --vendor-ffmpeg $(which ffmpeg) $(which ffprobe)
```

libx264가 포함된 ffmpeg 빌드는 GPL이다. 재배포하면 해당 바이너리에 대한
GPL 의무(라이선스 전문·소스 제공)를 지게 된다. 이 프로젝트 코드는 별도
프로세스로 호출하므로 어느 쪽이든 MIT를 유지한다.
