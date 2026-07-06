# wake-up

졸음 감지 프로그램. 두 개의 하위 프로젝트로 구성됩니다.

- [`drowsiness-detector/`](drowsiness-detector/) — 웹캠으로 눈을 캡처하고, 감김 상태가 일정 시간 지속되면 알람을 재생하는 데스크톱 클라이언트. 눈 상태 판정은 `eye-status-api`에 위임합니다.
- [`eye-status-api/`](eye-status-api/) — 눈 크롭 이미지를 받아 open/closed를 판정하는 무상태 추론 API 서버.

실행 방법은 각 폴더의 README를 참고하세요. `eye-status-api`를 먼저 띄운 뒤, `drowsiness-detector`의 `EYE_STATUS_API_URL` 환경변수를 그 주소로 맞추고 실행합니다.
