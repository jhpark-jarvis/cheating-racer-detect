# cheating-racer-detect

블랙박스 영상에서 차선 변경과 방향지시등 관측을 연결해 검토할 구간을 찾는 도구입니다.
현재는 지정 구간을 클립으로 추출하는 Windows CLI를 제공하며, 자동 탐지는 개발 예정입니다.

## 사용 기술

| 기술 | 역할 | 상태 |
| --- | --- | --- |
| Python | CLI·작업 제어·원본/결과 기록 | 사용 중 |
| FFmpeg / ffprobe | 영상 정보·시간축 분석, 디코딩·클립 추출 | 사용 중 |
| OpenCV | 프레임 처리, 가변 시간 간격 움직임 예측·검토 오버레이 | 연결·표시 모듈 합성 검증 |
| PyTorch | 차량·차선 분석 모델의 추론 기반 | 차량 탐지 합성 검증 |

차량 탐지는 YOLOX-s를 실험 중이며 최종 모델은 미정입니다. 모델 독립 탐지·추적 연결, 독립 ID 관리와 검토 오버레이를 라이브러리로 제공합니다.

## 구현 방향

원본 프레임 시각(PTS)을 유지하며 다음 분석 흐름을 구현할 계획입니다.

1. 차량 탐지·추적과 차선 상대 위치로 차선 변경 시점·방향을 찾습니다.
2. 후보 전후의 원본 차량 ROI를 시계열로 분석해 방향지시등 점멸을 관측합니다.
3. 가림·해상도 부족은 판정 불가로 표시하고, 사용자가 원본 맥락을 검토한 뒤 클립으로 저장합니다.

## 설치 및 사용

검증 환경: Windows · Python **3.13.13** · FFmpeg/ffprobe **9.0.2 Gyan essentials**. GPU는 필요하지 않습니다.
Python 설치 후 프로젝트 루트의 PowerShell에서 실행합니다.

```powershell
pwsh -NoProfile -File scripts/bootstrap.ps1 -DownloadTools

.\.venv\Scripts\python.exe -m cheating_racer_detect doctor
.\.venv\Scripts\python.exe -m cheating_racer_detect clip --input "D:\dashcam\input.mp4" --start 12.35 --end 19.8 --output "D:\dashcam\review-001"
```

예시 경로를 자신의 파일로 바꾸세요. 출력 상위 폴더는 존재해야 하며 결과 폴더는 새 이름이어야 합니다.
[설치 스크립트](scripts/bootstrap.ps1)는 로컬 환경을 준비하고 FFmpeg를 다운로드해 고정 SHA-256으로 검증합니다. 전역 Python·PATH·드라이버는 변경하지 않습니다.
기존 도구를 사용하려면 `--ffmpeg-dir` 또는 `CRD_FFMPEG_DIR`를 지정하세요. 명령별 옵션은 `--help`로 확인할 수 있습니다.

## 지원 범위

| 항목 | 내용 |
| --- | --- |
| 입력 | 단일 MP4 · H.264 8-bit 4:2:0 SDR 비디오 1개 · AAC 오디오 0~1개 |
| 시간 | 첫 비디오 프레임을 0초로 한 `[start,end)` · CFR/VFR·시작 PTS 오프셋·90도 단위 회전 |
| 출력 | 재인코딩한 `clip.mp4` + 원본 해시·요청/실제 시각·검증 이력을 담은 `result.json` |
| 종료 | 성공 `0` · JSON 오류 `2` · Ctrl+C/Ctrl+Break 취소 `CANCELLED`/`130` |

원본과 기존 출력은 덮어쓰지 않으며 영상은 로컬에서 처리합니다. 실제 프레임 경계에 따른 시각 차이는 결과에 기록됩니다.
HEVC·HDR·데이터/자막/다중 영상 트랙·분할 파일 결합 및 네트워크 경로·재분석 지점·상위 경로(`..`)는 지원하지 않습니다.
강제 종료 후 잔재가 남으면 해당 작업의 `.crd-*.partial`만 확인·정리하고 새 출력 이름으로 재시도하세요.

## 개발 및 검증

[추적 모듈](cheating_racer_detect/tracking/tracker.py)은 원본 PTS의 실제 시간 간격, 클래스별 관측 연결, 짧은 가림·초 단위 만료를 처리합니다.
모호한 연결은 이력을 끊고 예측과 실제 관측을 구분합니다. 자동 영상 분석 명령은 아직 제공하지 않습니다.
`Tracker.update(frame, detections)`에는 원본 시각을 담은 `Frame`, 같은 프레임의 `Detection`, 명시적인 `Policy`가 필요합니다.
좌표는 원본 coded raster의 half-open xyxy이며 회전은 메타데이터로 유지합니다. 정책 수치는 실제 차량 평가로 정해야 합니다.

탐지·추적 연결과 오버레이 렌더링에는 NumPy/OpenCV가 필요합니다. 별도 Python 환경에서 `python -m pip install ".[tracking]"`으로 준비할 수 있으며 기존 클립 CLI에는 필요하지 않습니다. 계약·검토 요약은 표준 라이브러리만 사용합니다.

### 분석 라이브러리

[탐지 연결](cheating_racer_detect/analysis/bridge.py)은 호출자가 제공한 탐지기를 사용합니다. `detector.detect(frame, image)`는 현재 `Frame`·원본 좌표의 `DetectorObservation`·`Letterbox`·모델 식별자를 담은 `DetectorBatch`를 반환해야 합니다. 입력은 같은 디코딩에서 얻은 BGR bytes, 탐지기에 전달되는 배열은 읽기 전용입니다. 현재 연결은 축 1024 이하·640×640 letterbox·프레임당 탐지 64개 이하로 제한합니다.

```python
from cheating_racer_detect.analysis import DetectorTrackerBridge
from cheating_racer_detect.tracking import Tracker
from cheating_racer_detect.review import render

# detector, policy, frame, bgr_bytes는 호출자가 준비합니다.
bridge = DetectorTrackerBridge(detector, Tracker(policy),
    model_alias="vehicle-model-v1", expected_detector_id=detector.model_id)
result = bridge.update(frame, bgr_bytes)
preview_bgr, review = render(frame, bgr_bytes, result["tracking"])
```

모델 공간의 YOLOX 형식 Nx7 결과에는 `post_nms_observations`를 한 번 적용합니다. 이미 원본 좌표인 박스에는 적용하지 않습니다. 모델 로딩·다운로드·영상 디코딩·파일 저장은 호출자 책임이며 자동으로 수행하지 않습니다.

[검토 오버레이](cheating_racer_detect/review/overlay.py)는 실제 관측을 실선, 위치 예측을 점선, 모호한 탐지를 ID 없이 표시합니다. 원본 시각과 개별 관측 목록을 함께 제공하며, 화면은 원본 coded raster 기준이고 회전 정보는 적용하지 않고 표시합니다. ID는 검증된 신원이 아니며 예측·모델 점수는 점멸 근거나 위반 확률이 아닙니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 자체 합성 영상을 사용합니다. FFmpeg가 없으면 통합 테스트가 건너뛰어지므로 전체 검증으로 볼 수 없습니다.
NumPy/OpenCV가 없으면 실제 추적 엔진 테스트도 건너뛰어집니다. 순수 계약 테스트와 실제 엔진 검증은 구분해야 합니다.
수동 확인용 합성 자료 생성은 [QA 도구](scripts/prepare_manual_qa.py)의 `--help`를 참고하세요.
실영상 적합성·수동 재생·장시간 성능·패키지 배포는 검증이 남아 있으며, 현재 처리 시간은 원본 전체 길이의 영향을 받습니다.

프로젝트 라이선스는 미정입니다.
