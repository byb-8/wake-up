# Eye Status API

눈 크롭 이미지(좌/우) 2장을 받아 open/closed 상태를 판정하는 무상태(stateless) 추론 API입니다.
`dima806/closed_eyes_image_detection` (ViT) 모델을 사용합니다.

## 실행 방법

```bash
uv sync --frozen
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

- 최초 실행 시 HuggingFace에서 모델을 다운로드합니다 (~340MB, 인터넷 연결 필요).
- CUDA GPU가 있으면 자동으로 사용하고, 없으면 CPU로 동작합니다 (별도 설정 불필요).
- `--host 0.0.0.0`으로 띄워야 외부에서 접근 가능합니다. 로컬 테스트만 할 땐 `127.0.0.1`도 가능.

## API

### `GET /health`

```json
{"status": "ok", "device": "cpu"}
```

### `POST /v1/eye-status`

`multipart/form-data`로 `left_eye`, `right_eye` 이미지 파일(JPEG/PNG 등) 전송.

```json
{
  "left_closed_prob": 0.12,
  "right_closed_prob": 0.09,
  "avg_closed_prob": 0.10,
  "is_closed": false,
  "threshold": 0.6,
  "device": "cpu"
}
```

잘못된 이미지(디코딩 불가)를 보내면 400을 반환합니다.

## 배포 시 참고사항

- **워커 수**: `uvicorn`에 `--workers N`을 주면 워커별로 모델을 각자 메모리에 로드합니다 (N배 메모리 사용). 동시 요청이 많지 않다면 워커 1개(기본값)로 충분합니다. 늘릴 경우 서버 메모리 여유를 확인하세요.
- **동시 요청 처리**: 추론 자체는 스레드풀에서 실행되므로, 워커 1개로도 여러 요청이 순차 대기 없이 어느 정도 겹쳐서 처리됩니다.
- **모델 다운로드**: 컨테이너/서버가 매번 새로 뜨는 환경(예: 서버리스)이라면 시작할 때마다 재다운로드가 발생하니, 모델을 이미지에 미리 포함하거나 캐시 볼륨을 유지하는 걸 권장합니다. HuggingFace Hub 요청이 잦으면 `HF_TOKEN` 환경변수 설정을 권장합니다 (레이트 리밋 완화).
- **CORS**: 브라우저에서 직접 호출할 계획이 있다면 CORS 설정이 필요합니다 (현재는 없음). 서버-to-서버 호출(현재 클라이언트인 `drowsiness-detector`)은 CORS와 무관합니다.
