# cheating-racer-detect

블랙박스 영상에서 차선 변경 차량을 찾고, 변경 전후 방향지시등 관측을 연결해 사람이 확인할 구간을 줄이는 프로젝트다. 이름은 현재 폴더명이며 정식 제품명은 미정이다.

현재는 **Windows 로컬 클립 생성 CLI의 기반 구현**이 있다. 영상과 시작·끝 시간을 받아 `clip.mp4`와 출처·검증 기록 `result.json`을 만든다. 차량·차선·방향지시등 자동 탐지와 검토 화면은 아직 구현하지 않았다.

## 설치·실행

프로젝트 루트의 PowerShell에서 실행한다. 현재 검증한 조합은 Windows, Python **3.13.13**, FFmpeg/ffprobe **9.0.2 Gyan essentials**다. 첫 CLI는 Python 표준 라이브러리만 사용하며 GPU가 필요하지 않다.

```powershell
# 최초 준비: Python 3.13은 설치되어 있어야 한다.
# -DownloadTools는 공개 FFmpeg 배포 파일을 내려받고 고정 SHA-256으로 확인한다.
pwsh -NoProfile -File scripts/bootstrap.ps1 -DownloadTools

.\.venv\Scripts\python.exe -m cheating_racer_detect doctor
.\.venv\Scripts\python.exe -m cheating_racer_detect clip --input "D:\dashcam\input.mp4" --start 12.35 --end 19.8 --output "D:\dashcam\review-001"
```

예시 영상 경로는 자신의 파일로 바꾼다. 출력의 상위 폴더는 이미 존재해야 하고 결과 폴더 `review-001`은 없어야 한다. 원본이나 기존 결과는 덮어쓰지 않는다. `--help`로 인자를 확인한다. 필요하면 각 명령에 `--ffmpeg-dir "도구의 bin 경로"`를 주거나 `CRD_FFMPEG_DIR`를 설정할 수 있다. 우선순위는 명시 인자 → 환경 변수 → 프로젝트 도구 → PATH다.

설치 범위는 `.venv/`와 `.tools/`이며 전역 Python·PATH·드라이버를 바꾸지 않는다. FFmpeg 다운로드/압축 해제는 [bootstrap](scripts/bootstrap.ps1)의 고정 체크섬과 기존 폴더 비덮어쓰기 규칙을 사용한다. 서버로 영상이나 결과를 보내지 않는다. 패키지 배포용 `crd` 진입점은 메타데이터에 선언했지만 현재 검증된 실행 방법은 위 `python -m` 방식이다.

지원하는 첫 프로파일은 **자체 데이터를 포함한 단일 MP4, H.264 8-bit 4:2:0 SDR 비디오 1개, AAC 오디오 0~1개**다. CFR/VFR, 0이 아닌 시작 PTS, 90도 단위 회전을 합성 영상으로 검증한다. HEVC·HDR·데이터/자막/다중 영상 트랙·분할 파일 결합은 아직 지원하지 않는다. 로컬 파일만 받으며 UNC·네트워크 드라이브·재분석 지점·상위 경로(`..`)는 거부한다.

시간은 첫 비디오 프레임을 0초로 한 `[start,end)`이며, 그 구간 안에서 표시를 시작하는 프레임을 포함한다. 첫 포함 프레임이 요청 시작보다 늦거나 마지막 프레임의 표시 끝이 요청 끝보다 늦을 수 있어 실제 PTS와 경고를 결과에 남긴다. 재인코딩한 검토용 파생물이며 원본 바이트와 같지 않다.

오류는 표준 오류에 JSON 코드/메시지와 종료 코드 2, 취소는 `CANCELLED`와 130으로 전달한다. Ctrl+C/Ctrl+Break를 처리한다. 처리 중에는 `.crd-*.partial`만 만들고 검증 뒤 최종 디렉터리로 확정한다. 강제 종료로 잔재가 남으면 해당 작업 소유인지 확인하고 정리한 뒤 새 출력 이름으로 다시 실행한다. 프로그램은 다른 작업의 잔재를 자동 삭제하지 않는다.

분석 기반으로 Python, FFmpeg/ffprobe, OpenCV, PyTorch가 승인됐다. 첫 CLI의 입출력은 `argparse`·`subprocess`·`decimal/fractions`·`json`, 테스트는 `unittest`, 환경은 `venv`를 사용한다. 제3자 Python 런타임 의존성은 없다. OpenCV/PyTorch는 아직 설치하지 않았으며 모델별 종속성과 GPU 검증 후 버전을 고정한다. UI·DB·웹서버도 아직 채택하지 않았다.

## 개발·검증

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 외부 영상 다운로드 없이 합성 영상을 OS 임시 폴더에 만들고 정리한다. FFmpeg가 없으면 통합 테스트가 `SKIPPED / NOT_RUN`이므로 전체 기능 통과로 해석하지 않는다. 실제 블랙박스, 수동 플레이어 확인, AI·GPU·대용량 성능, 패키지 빌드/배포는 별도 검증이다. 기본 구현은 정확성 확인을 위해 원본 전체를 여러 번 읽고 디코딩하므로 짧은 구간 추출도 입력 전체 길이의 영향을 받는다.

진입점은 [CLI](cheating_racer_detect/cli.py), 트랜잭션과 원본 보존은 [service](cheating_racer_detect/service.py), 미디어 시간 처리·검증은 [media](cheating_racer_detect/media.py), 로컬 경계는 [paths](cheating_racer_detect/paths.py), 도구/프로세스 관리는 [tools](cheating_racer_detect/tools.py)다.

### 수동 재생·취소 확인

민감 정보가 없는 합성 검토 자료를 새 폴더에 생성할 수 있다. 기존 폴더는 덮어쓰지 않는다. `.artifacts` 상위 폴더를 먼저 준비하고 아래 명령을 코드 저장소 루트에서 실행한다.

```powershell
New-Item -ItemType Directory -Path .artifacts -Force | Out-Null
.\.venv\Scripts\python.exe -m scripts.prepare_manual_qa --output .artifacts/manual-qa --with-cancel
```

`fixtures/`의 원본과 각 `audio`, `cfr`, `rotation90`, `rotation180`, `rotation270` 폴더의 `clip.mp4`를 플레이어에서 비교한다. 재생·일시정지·탐색이 되고 회전된 표시 방향이 원본과 같아야 한다. `audio`는 오른쪽 흰색 flash와 beep가 약 1.15초에 함께 나오며, `cfr`는 음성 트랙이 없어야 한다. 결과는 요청 시작 0.35초부터 끝 2.35초 사이에 표시를 시작하는 프레임을 포함한다. 영상 길이만으로 프레임 정합성을 판단하지 않는다.

실제 키 입력 취소는 아래 명령 실행 중 Ctrl+C를 누른 후 `$LASTEXITCODE`가 130인지, `CANCELLED`가 나오는지, 최종 `cancel-result`와 이번 작업의 `.crd-*.partial`이 남지 않는지 확인한다. 같은 명령을 다시 실행해 정상 완료(exit 0, 영상+JSON)를 확인한다. 너무 빨리 완료되면 테스트를 PASS로 기록하지 말고 새 출력 이름으로 다시 시도한다.

```powershell
.\.venv\Scripts\python.exe -m cheating_racer_detect clip --input .artifacts/manual-qa/cancel-source.mp4 --start 0.35 --end 2.35 --output .artifacts/manual-qa/cancel-result
$LASTEXITCODE
```

합성 자료 생성 성공은 사람의 재생·소리·키 입력 확인을 의미하지 않는다. 플레이어/버전·관측 결과는 별도로 기록한다. 생성 자료는 자동 삭제하지 않으며 확인 후 해당 QA 폴더만 소유자가 정리한다. 디스크 부족/코덱 부재는 자동 테스트의 안전한 오류 주입으로 확인하며 실제 디스크를 채우거나 설치된 코덱을 삭제하지 않는다.

## 공개 준비 상태

현재는 로컬 개발 단계이며 자동 차량·차선·방향지시등 탐지는 미구현이다. 실영상·수동 재생·장시간 처리 성능·패키지 배포 검증이 남아 있다. 프로젝트 라이선스는 아직 정하지 않았으며 공개/재배포 전에 별도로 확정한다.
