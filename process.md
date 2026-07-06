다음 기능을 하는 Python 프로그램을 작성해줘.

## 목표
웹캠으로 실시간으로 사람 눈 상태를 감지해서, 5초 이상 눈을 감고 있으면
지정한 유튜브 영상을 재생해서 사용자를 깨우는 졸음 감지 프로그램

## 환경 설정 (uv 사용)
아래 순서대로 환경을 세팅해줘:

uv init drowsiness-detector
cd drowsiness-detector
uv add torch torchvision transformers opencv-python pillow pygame yt-dlp

## 사용할 기술 스택
- 눈 감지 모델: HuggingFace의 dima806/closed_eyes_image_detection (ViT 기반)
- GPU 사용: PyTorch CUDA (device=0)
- 웹캠 입력: OpenCV (cv2.VideoCapture)
- 영상 재생: yt-dlp로 유튜브 URL에서 스트림 URL 추출 후 pygame으로 재생 (소리 포함)

## 상세 동작 흐름
1. 프로그램 시작 시 알람으로 쓸 유튜브 URL을 변수로 지정 (ALARM_YOUTUBE_URL)
2. 프로그램 시작 시 yt-dlp로 해당 URL의 스트림 주소를 미리 추출해둠
3. 웹캠에서 실시간으로 프레임을 읽어옴
4. 매 프레임마다 HuggingFace 모델로 눈 상태(open/closed) 분류
5. 눈이 감긴 상태가 연속으로 5초 이상 지속되면 알람 영상 재생
6. 눈을 뜨면 알람 즉시 중지
7. 웹캠 화면에 현재 상태 오버레이 표시

## 화면 표시 요구사항
- 눈 뜬 상태: 초록색 텍스트로 "OPEN 👀"
- 눈 감은 상태: 빨간색 텍스트로 "CLOSED 😴 Xsec"
- 알람 재생 중: 화면 테두리를 빨간색으로 표시

## 코드 요구사항
- ALARM_YOUTUBE_URL 변수에 유튜브 링크 지정
- 프로그램 시작 시 yt-dlp로 스트림 URL 미리 추출
- 모델 로딩은 시작 시 한 번만
- 영상 재생은 별도 스레드로 처리 (웹캠 멈추지 않게)
- 'q' 키로 종료

## 실행 방법도 README.md로 작성해줘
uv run main.py 로 실행할 수 있게