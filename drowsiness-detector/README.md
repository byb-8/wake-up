# Drowsiness Detector

웹캠으로 눈 상태를 실시간 감지해, **5초 이상 눈을 감으면 YouTube 알람**을 재생하는 졸음 감지 프로그램입니다.

## 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| 눈 상태 분류 | HuggingFace `dima806/closed_eyes_image_detection` (ViT) |
| GPU 추론 | PyTorch CUDA (`cuda:0`, CPU fallback 지원) |
| 웹캠 입력 | OpenCV (`cv2.VideoCapture`) |
| 얼굴 감지 | OpenCV Haar Cascade |
| 스트림 URL 추출 | yt-dlp |
| 알람 재생 | pygame (음성 포함) |

## 실행 방법

### 1. 의존성 설치 (최초 1회)

```bash
uv init drowsiness-detector
cd drowsiness-detector
uv add torch torchvision transformers opencv-python pillow pygame yt-dlp
```

### 2. 알람 영상 설정

`main.py` 상단의 `ALARM_YOUTUBE_URL` 변수에 원하는 YouTube 링크를 입력하세요.

```python
ALARM_YOUTUBE_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

### 3. 프로그램 실행

```bash
uv run main.py
```

## 동작 흐름

1. 시작 시 yt-dlp로 YouTube 오디오 스트림 URL을 미리 추출
2. 백그라운드에서 오디오 데이터를 버퍼에 로드
3. HuggingFace ViT 모델을 GPU(또는 CPU)에 로드
4. 웹캠 스트림을 실시간으로 읽어 3프레임마다 눈 상태 분류
5. 연속 5초 이상 눈 감힘 → 알람 재생
6. 눈을 뜨면 알람 즉시 중지

## 화면 표시

| 상태 | 표시 |
|------|------|
| 눈 뜸 | 초록색 `OPEN` 텍스트 |
| 눈 감음 | 빨간색 `CLOSED Xs` 텍스트 |
| 알람 재생 중 | 빨간색 테두리 |
| 얼굴 감지 | 노란색 얼굴 박스 |

## 조작키

- `q` — 프로그램 종료

## 참고사항

- CUDA GPU가 없으면 CPU로 자동 전환됩니다 (추론 속도가 느릴 수 있음).
- 처음 실행 시 HuggingFace에서 모델을 다운로드합니다 (~340MB).
- 인터넷 연결이 필요합니다 (모델 다운로드 + YouTube 스트림 URL 추출).
