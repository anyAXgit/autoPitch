# autoPitch

[![CI](https://github.com/anyAXgit/autoPitch/actions/workflows/ci.yml/badge.svg)](https://github.com/anyAXgit/autoPitch/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/anyAXgit/autoPitch)](https://github.com/anyAXgit/autoPitch/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

두 시간짜리 풋살 영상에서 하이라이트 후보를 뽑아 컷으로 만들어 줍니다.
다 돌려볼 필요 없이, 나온 후보를 확인해서 넣고 빼면 됩니다.

소리로 찾습니다. 골이 들어가면 사람들이 반응하고 그 순간 오디오가 튑니다 —
화면을 볼 필요가 없습니다. 여기에 고정 카메라라는 공짜 정보를 얹어(골대 위치가
늘 같으니) 컷 시작점을 프레임 단위로 맞춥니다.

## 기능

- **두 카메라 시각 맞추기** — 현장음 상호상관으로 자동. 메타데이터 없이 맞습니다.
- **반응 지연 보정** — 두 구장 140골 이상에서 소리는 골보다 중앙값 1.5초 늦게
  터집니다. 그만큼 앞으로 당겨 자릅니다.
- **골 난 쪽 앵글 선택** — 골망의 모션 스파이크로 어느 쪽에서 났는지 알고 그
  카메라를 씁니다. 스파이크가 약하면 함성 기준으로 폴백합니다.
- **브라우저 편집기** — 후보를 실제 영상으로 스크럽하며 컷 지점·앵글을 다듬고
  바로 렌더합니다.
- **골 라벨링** (선택) — 후보 프레임을 VLM 에 보내 골 여부를 데이터로 기록합니다.
- **로컬 전용** — 촬영본이 어디로도 올라가지 않습니다. 라벨링을 켤 때만 정지
  프레임 몇 장이 API 로 갑니다.
- **거창한 장비 불필요** — 액션캠·휴대폰 한두 대를 펜스에 물리면 됩니다.

## 요구 사항

| | |
|---|---|
| **ffmpeg** | 필수. 없으면 첫 화면이 알려주고 버튼 하나로 설치합니다 |
| **Python 3.10+** | 소스로 돌릴 때만 — 설치판은 파이썬 없이 동작합니다 |
| **Anthropic API 키** | 선택. 골 라벨링을 켤 때만 |

## 설치

### 설치판

[Releases](https://github.com/anyAXgit/autoPitch/releases)에서 macOS / Windows
빌드를 받습니다. 실행하면 홈에 `~/autoPitch` 작업 공간이 생기고 브라우저가
열립니다. 첫 화면이 환경을 점검합니다.

> macOS 빌드는 Developer ID 로 서명·공증돼 있어 그냥 열립니다.
> Windows 빌드는 아직 서명 전이라 SmartScreen 경고가 뜹니다 —
> `추가 정보` → `실행` 을 눌러 주세요.

### 소스에서

```bash
git clone https://github.com/anyAXgit/autoPitch.git
cd autoPitch
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python autopitch.py          # 브라우저 스튜디오
```

ffmpeg 는 따로 설치합니다:

```bash
brew install ffmpeg                      # macOS
winget install --id Gyan.FFmpeg          # Windows
apt install ffmpeg                       # Debian/Ubuntu
```

## 사용법

1. **카메라별로 촬영본이 든 폴더를 지정합니다.** 복사하지 않고 그 자리에서
   읽습니다. 지정하지 않으면 작업 폴더의 `data/raw/<날짜>/cam1`, `cam2` … 를
   씁니다 (단일캠이면 `cam1` 만).
2. **골대 위치를 한 번 그립니다.** 건너뛰어도 하이라이트는 나옵니다.
3. **`장면 찾기`** → 나온 후보를 확인·수정 → 내보내기.

카메라는 **대각선으로 마주 보는 코너**에 두고 경기 중에는 움직이지 않게
고정합니다. 설정 화면의 *카메라 어떻게 놓을까* 에 그림과 함께 정리해 두었습니다.

명령줄로도 됩니다:

```bash
./.venv/bin/python main.py                # data/raw 전체
./.venv/bin/python run_sessions.py --all  # 날짜별 여러 경기
```

## 설정

`config.yaml` 에서 임계값(`threshold_k`), 길이(`min_len_sec`/`max_len_sec`),
피크 상한(`max_clips`), 크로스페이드, 골 후 마진(`post_goal_sec`) 등을 조정합니다.
자주 만지는 값은 GUI 의 `세부 설정` 에도 있습니다. 거기서 저장해도 이 파일의
주석과 정렬은 그대로 남습니다 — 왜 그 값인지가 주석에 적혀 있어서입니다.

### 배경음악

`config.yaml` 에 `bgm_path: data/bgm.mp3` 를 지정하면 `highlight_all.mp4` 에
음악을 깔아줍니다. 관중 소리는 그대로 두고 음악만 `bgm_volume` 으로 낮게 믹싱하며,
길이에 맞춰 루프합니다.

### 골 라벨링 (선택)

`vision.enabled: true` 면 후보 주변 프레임 몇 장을 VLM 에 보내 골인지 물어보고,
그 답을 `vision_goal` · `vision_conf` 로 **기록만 합니다.** 클립을 지우지도, 배지를
그리지도 않습니다 — 모델도 틀릴 수 있어서 최종 판단은 편집기에서 사람이 합니다.

모델은 `vision.model`(기본 `claude-sonnet-5`, `claude-haiku-4-5` 가 저렴),
기준은 `vision.min_confidence` 로 조정합니다.

```bash
./.venv/bin/pip install anthropic
export ANTHROPIC_API_KEY=...     # 또는 작업 폴더의 data/_gui/anthropic_key.txt
```

### 네트-ROI 골 정밀 위치 (선택, 무API)

카메라가 고정이면 골망 위치도 고정입니다. 공이 네트를 때리는 **모션 스파이크**로
정확한 골 프레임을 잡고, 어느 쪽 네트에서 났는지로 **앵글까지** 고릅니다.

캘리브레이션은 **각 카메라에 가까운 골망 하나만** 박스로 잡으면 됩니다(먼 골은
반대편 카메라가 맡습니다) → `net_rois.json`. `locate.enabled: true` 로 켭니다.
골망이 안 보이거나 스파이크가 약하면 자동으로 함성 onset 으로 폴백합니다.

### 조용한 골 스캔 (기본 꺼짐)

반응이 없는 골을 골대 영역의 움직임만으로 찾아보려는 실험입니다.
`locate.scan_enabled` 로 켜지만 **기본값은 꺼짐**입니다 — 개발에 쓴 카메라
배치(측면 낮은 코너)에서는 골과 잡음이 갈리지 않았습니다. 시도한 방법과 결과는
`config.yaml` 주석에 적어 두었습니다.

켜면 남은 후보가 편집기에 **ROI?** 배지로 떠서 직접 확인·제외할 수 있고, 판정은
`data/train_events.jsonl` 에 쌓입니다. 카메라를 골대 뒤에 두는 배치라면 다시 볼
만합니다.

## 동작 방식

GUI 는 파이썬 stdlib 로 된 로컬 서버입니다. 원본을 range 스트리밍해서 브라우저에서
실제 영상을 스크럽하며 컷을 다듬고, 그대로 렌더까지 합니다. 탭은 ① 설정 ·
② 골 ROI · ③ 편집·렌더 입니다.

```bash
./.venv/bin/python gui/server.py        # http://127.0.0.1:8756
```

서버는 `127.0.0.1` 에만 붙고 **자기 페이지에서 온 요청만** 처리합니다
(`Host`·`Origin`·`Sec-Fetch-Site` 확인). 루프백 바인드만으로는 다른 기계를 막을
뿐 브라우저의 다른 탭을 못 막기 때문입니다.

편집 결정(컷 in/out·앵글)은 `plan.json` 에 데이터로 남습니다. GUI 없이
`editor/index.html` 로 열어 고친 뒤 다시 렌더할 수도 있습니다:

```bash
./.venv/bin/python render_from_plan.py plan.edited.json data/output/edited
```

## 한계

실측한 것만 적습니다.

- **아무도 반응하지 않은 장면은 찾지 못합니다.** 소리가 기준이라 그렇습니다.
  카메라를 골대 뒤에 하나 더 두면 달라질 수 있고, 그건 아직 해보지 않았습니다.
- **골이 아닌 장면도 올라옵니다.** 버그가 아니라 의도입니다 — 선방·아쉬운 슛도
  후보로 올리고, 넣고 빼는 건 편집기에서 합니다.
- **다른 구장**은 골대 위치만 다시 그리면 그대로 동작합니다.

## 개발

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest -q          # 실제 영상 없이 합성 더미로 검증
./.venv/bin/python packaging/build.py --clean
```

결과는 `dist/` 에 생깁니다 (macOS `autoPitch.app`, Windows `autoPitch.exe`).
PyInstaller 는 크로스 컴파일을 못 하므로 각 OS 에서 따로 빌드해야 합니다.

서명·공증까지 한 번에 (macOS, Developer ID 인증서 필요):

```bash
./.venv/bin/python packaging/build.py --clean --sign --notarize
```

`--sign` 은 설치된 Developer ID 인증서를 자동으로 찾고, `--notarize` 는 키체인
프로필(`autopitch-notary`)이나 App Store Connect 키(`AUTOPITCH_NOTARY_KEY` 등
환경변수)로 인증합니다. 태그를 밀면 CI 가 같은 일을 합니다.

<details>
<summary><b>ffmpeg 를 왜 번들하지 않는가</b></summary>

**결정: 배포 빌드에 ffmpeg 를 넣지 않는다.** 첫 화면이 유무를 확인하고, 없으면
`ffmpeg 설치하기` 버튼 한 번으로 각 OS 의 패키지 관리자로 설치한다.

측정해 보면 파이프라인이 요구하는 것 중 **GPL 인 건 `libx264` 하나뿐**이다.
aac 인코딩, xfade/acrossfade, overlay/scale/pad, 하드웨어 인코더
(videotoolbox·nvenc·qsv·amf)는 전부 LGPL 범위다. 실제로 하드웨어 인코더만으로
전체 렌더가 완결되는 것을 확인했다.

그래서 세 갈래가 있었다:

| | HW 인코더 있는 기기 | HW 인코더 없는 기기 | 라이선스 부담 |
|---|---|---|---|
| LGPL 번들 (x264 제외) | 동작 | **렌더 불가** | LGPL 고지·소스 제공 |
| GPL 번들 (x264 포함) | 동작 | 동작 | **GPL 소스 제공 의무** |
| 번들 안 함 (채택) | 동작 | 동작 | 없음 |

LGPL 번들은 "느려짐" 이 아니라 **아예 못 쓰는 기기**를 만든다(GitHub 의 Windows
러너가 정확히 그런 환경이다). openh264 를 넣으면 해결되지만 macOS 용은 직접
빌드해야 하고 특허 관계도 단순하지 않다. GPL 번들은 동작하지만 MIT 프로젝트의
릴리스에 GPL 바이너리를 넣고 그 소스 제공 의무를 지는 일이다.

서명 여부와도 무관하다. macOS 빌드는 Developer ID 로 서명·공증되니 그냥 열리고,
그건 ffmpeg 를 넣든 말든 같다. Windows 는 아직 미서명이라 어차피 경고를 한 번
거치므로, 번들해서 "더블클릭하면 그냥 됨" 을 사는 것도 아니다.

직접 번들하려면:

```bash
./.venv/bin/python packaging/build.py --vendor-ffmpeg $(which ffmpeg) $(which ffprobe)
```

libx264 가 포함된 ffmpeg 빌드는 GPL 이다. 재배포하면 해당 바이너리에 대한 GPL
의무(라이선스 전문·소스 제공)를 지게 된다. 이 프로젝트 코드는 별도 프로세스로
호출하므로 어느 쪽이든 MIT 를 유지한다.

</details>

## 기여

버그 신고와 다른 구장에서의 결과 보고를 환영합니다.
먼저 [CONTRIBUTING.md](CONTRIBUTING.md) 를 봐 주세요.
이슈 템플릿에 **"다른 구장에서 돌려봤습니다"** 가 따로 있습니다 — 어떤 배치에서
되고 안 되는지가 이 프로젝트에 제일 필요한 데이터입니다.

보안 취약점은 공개 이슈 대신 [SECURITY.md](SECURITY.md) 의 절차로 알려주세요.

변경 내역은 [CHANGELOG.md](CHANGELOG.md).

## 라이선스

[MIT](LICENSE) © anyax

- 코어는 numpy · scipy · Pillow · PyYAML 뿐이고 전부 허용형(BSD/MIT)입니다.
- **ffmpeg** 은 번들하지 않고 별도 프로세스로 호출합니다 — 라이선스가 전이되지
  않습니다.
- **`src/player_analysis.py`**, **`tools/kickoff_scan.py`** 는 선택·실험 모듈로
  `ultralytics`(**AGPL-3.0**)를 씁니다. 코어와 배포 앱은 이를 임포트하지 않고
  `requirements.txt` 에도 없습니다. 직접 설치해 쓰는 경우 그 의무는 사용자에게
  있습니다.
