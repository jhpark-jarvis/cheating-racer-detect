# cheating-racer-detect

블랙박스 영상의 지정 구간을 검토용 클립으로 추출하는 Windows CLI입니다.
차선 변경·방향지시등 관측을 통한 검토 후보 자동 탐지는 개발 예정입니다.

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

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 자체 합성 영상을 사용합니다. FFmpeg가 없으면 통합 테스트가 건너뛰어지므로 전체 검증으로 볼 수 없습니다.
수동 확인용 합성 자료 생성은 [QA 도구](scripts/prepare_manual_qa.py)의 `--help`를 참고하세요.
실영상 적합성·수동 재생·장시간 성능·패키지 배포는 검증이 남아 있으며, 현재 처리 시간은 원본 전체 길이의 영향을 받습니다.

프로젝트 라이선스는 미정입니다.
