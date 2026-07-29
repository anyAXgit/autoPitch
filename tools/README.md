# tools/

두 갈래가 섞여 있습니다.

**영상 제작용** — `make_graphic_*.py`, `make_outro_*.py`, `append_outro.py`.
이 프로젝트를 소개하는 유튜브 영상의 b-roll 그래픽을 만드는 스크립트입니다.
제품 코드가 아니며, 작성자의 촬영본 경로를 그대로 참조하는 것들이 있어
그대로는 다른 환경에서 돌지 않습니다. 어떻게 만들었는지 남겨두는 용도입니다.

```bash
pip install -r requirements-tools.txt   # matplotlib
```

**측정·실험** — `ball_scan.py`, `ball_in_net.py`, `kickoff_scan.py`.
조용한 골을 잡아보려던 시도들의 기록입니다. 대부분 측정 결과 실패로 결론났고
(README 의 "어디까지 되는가" 참고), 다른 카메라 배치에서 다시 볼 가치가 있어
남겨두었습니다. `kickoff_scan.py` 는 **AGPL-3.0** 인 ultralytics 를 씁니다 —
파일 상단 고지를 확인하세요.

어느 쪽도 코어 파이프라인이나 배포되는 앱이 임포트하지 않습니다.
