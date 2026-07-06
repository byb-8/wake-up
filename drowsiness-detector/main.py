#!/usr/bin/env python3
"""
Drowsiness Detector — webcam 기반 졸음 감지 프로그램
5초 이상 눈을 감고 있으면 YouTube 알람을 재생합니다.

눈 상태 분류는 로컬 모델이 아니라 eye-status-api 서버에 위임합니다.
"""

import cv2
import os
import queue
import sys
import tempfile
import threading
import time
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
import numpy as np
import pygame
import requests
import yt_dlp

# ─── 설정 ────────────────────────────────────────────────────────────────────

ALARM_YOUTUBE_URL = "https://www.youtube.com/shorts/FZnYTlPcgVY"
CLOSED_EYE_THRESHOLD = 2.0   # 눈 감은 상태 임계값 (초)
OPEN_FORGIVENESS = 6         # "open" 판정이 이 횟수 연속으로 나와야 타이머 리셋
CLOSED_CONFIDENCE = 0.60     # 두 눈 평균 closed 확률이 이 값 이상이면 감긴 것으로 판정 (표시용, 실제 판정은 API가 함)
EYE_STATUS_API_URL = os.environ.get("EYE_STATUS_API_URL", "http://127.0.0.1:8123/v1/eye-status")
API_TIMEOUT = 2.0            # API 응답 대기 최대 시간 (초)
INFER_EVERY_N_FRAMES = 3     # N 프레임마다 한 번 추론

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
EYE_CROP_PAD  = 20

FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ─── 전역 상태 ───────────────────────────────────────────────────────────────

_alarm_playing = False
_alarm_lock = threading.Lock()
_audio_path = None   # 다운로드된 mp3 파일 경로
_audio_ready = False

_frame_q = queue.Queue(maxsize=1)
_result_q = queue.Queue(maxsize=1)
_alarm_frame_q = queue.Queue(maxsize=2)  # 알람 영상 프레임 (메인 스레드에서 표시)

ALARM_WINDOW = "Alarm"


# ─── 알람 제어 ───────────────────────────────────────────────────────────────

def _download_alarm_audio(youtube_url):
    """백그라운드: yt-dlp로 원본 오디오 다운로드 → imageio_ffmpeg으로 mp3 변환."""
    import glob
    import subprocess
    global _audio_path, _audio_ready
    tmp = os.path.join(tempfile.gettempdir(), "drowsiness_alarm")
    mp3_path = tmp + ".mp3"
    try:
        print("🎵 알람 오디오 다운로드 중...")
        # ffmpeg 없이 원본 포맷으로 다운로드
        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
            "outtmpl": tmp + ".%(ext)s",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            ext = info.get("ext", "")

        raw_path = tmp + "." + ext if ext else None
        if not raw_path or not os.path.exists(raw_path):
            # 확장자를 모를 때 glob으로 탐색
            candidates = [f for f in glob.glob(tmp + ".*") if not f.endswith(".mp3")]
            raw_path = candidates[0] if candidates else None
        if not raw_path:
            raise FileNotFoundError("다운로드된 오디오 파일을 찾을 수 없습니다.")

        # imageio_ffmpeg 번들 바이너리로 mp3 변환
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", raw_path, "-vn",
             "-acodec", "libmp3lame", "-q:a", "4", mp3_path],
            capture_output=True, check=True,
        )

        try:
            os.remove(raw_path)
        except Exception:
            pass

        if os.path.exists(mp3_path):
            _audio_path = mp3_path
            _audio_ready = True
            print("✅ 오디오 준비 완료")
    except Exception as e:
        print(f"⚠️  오디오 다운로드 실패: {e}")


def _get_streams(youtube_url):
    """yt-dlp로 영상 URL과 오디오 URL을 한 번에 추출합니다."""
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

    fmts = info.get("formats", [info])

    # 단일 merged URL
    if "url" in info:
        return info["url"], info["url"]

    video_url = audio_url = None
    for fmt in reversed(fmts):
        if not video_url and fmt.get("vcodec") not in (None, "none"):
            video_url = fmt.get("url")
        if not audio_url and fmt.get("acodec") not in (None, "none") and fmt.get("vcodec") in (None, "none"):
            audio_url = fmt.get("url")
        if video_url and audio_url:
            break

    return video_url, (audio_url or video_url)


def _play_video_loop(video_url):
    """별도 스레드: 알람 영상 프레임을 읽어 큐에 넣습니다 (cv2 GUI 호출 없음)."""
    global _alarm_playing

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("⚠️  영상 스트림을 열 수 없습니다.")
        with _alarm_lock:
            _alarm_playing = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay = 1.0 / fps

    try:
        while _alarm_playing:
            ret, vframe = cap.read()
            if not ret:
                # 네트워크 스트림은 seek 불가 — 잠시 대기 후 재시도
                time.sleep(0.05)
                continue
            h, w = vframe.shape[:2]
            scale = min(854 / w, 480 / h)
            vframe = cv2.resize(vframe, (int(w * scale), int(h * scale)))
            try:
                _alarm_frame_q.get_nowait()
            except queue.Empty:
                pass
            _alarm_frame_q.put(vframe)
            time.sleep(frame_delay)
    finally:
        try:
            cap.release()
        except Exception:
            pass


def play_alarm():
    """영상+오디오 알람을 재생합니다 (별도 스레드에서 호출)."""
    global _alarm_playing
    with _alarm_lock:
        if _alarm_playing:
            return
        _alarm_playing = True

    try:
        print("🔗 알람 스트림 추출 중...")
        video_url, _ = _get_streams(ALARM_YOUTUBE_URL)

        if not video_url:
            raise RuntimeError("영상 스트림 URL을 찾을 수 없습니다.")

        # 오디오: pygame mixer (mp3 파일)
        if _audio_ready and _audio_path:
            pygame.mixer.music.load(_audio_path)
            pygame.mixer.music.play(-1)

        # 영상: cv2 스레드
        threading.Thread(target=_play_video_loop, args=(video_url,), daemon=True).start()
        print("🔔 알람 재생 시작")

    except Exception as e:
        print(f"⚠️  알람 재생 오류: {e}")
        with _alarm_lock:
            _alarm_playing = False


def stop_alarm():
    """알람(영상+오디오)을 즉시 정지합니다. cv2 창/VideoCapture 정리는 각자 담당."""
    global _alarm_playing
    with _alarm_lock:
        if not _alarm_playing:
            return
        _alarm_playing = False

    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

    print("🔕 알람 정지")


# ─── 모델 추론 ───────────────────────────────────────────────────────────────

def ensure_face_model():
    """face_landmarker.task 파일이 없으면 자동 다운로드합니다."""
    if not os.path.exists(FACE_MODEL_PATH):
        print("📥 FaceLandmarker 모델 다운로드 중...")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        print("✅ 다운로드 완료")


def _check_api():
    """eye-status-api 서버가 응답하는지 확인합니다."""
    print(f"🌐 eye-status-api 확인 중... ({EYE_STATUS_API_URL})")
    resp = requests.get(EYE_STATUS_API_URL.replace("/v1/eye-status", "/health"), timeout=API_TIMEOUT)
    resp.raise_for_status()
    print(f"✅ API 연결 성공: {resp.json()}")


def _query_eye_status(left_eye, right_eye):
    """좌/우 눈 크롭 이미지를 API로 보내 (is_closed, avg_closed_prob)를 반환합니다."""
    ok, left_buf = cv2.imencode(".jpg", left_eye)
    ok2, right_buf = cv2.imencode(".jpg", right_eye)
    if not ok or not ok2:
        raise ValueError("눈 이미지 인코딩 실패")
    files = {
        "left_eye": ("left.jpg", left_buf.tobytes(), "image/jpeg"),
        "right_eye": ("right.jpg", right_buf.tobytes(), "image/jpeg"),
    }
    resp = requests.post(EYE_STATUS_API_URL, files=files, timeout=API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["is_closed"], data["avg_closed_prob"]


def _inference_worker():
    """추론 워커 스레드: eye-status-api를 호출해 눈 상태를 판정합니다."""
    while True:
        try:
            item = _frame_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:  # 종료 신호
            break
        try:
            left_eye, right_eye = item
            is_closed, avg = _query_eye_status(left_eye, right_eye)
            try:
                _result_q.get_nowait()
            except queue.Empty:
                pass
            _result_q.put((is_closed, avg))
        except requests.RequestException as e:
            print(f"⚠️ eye-status-api 호출 실패: {e}")
        except Exception as e:
            print(f"추론 오류: {e}")


# ─── 화면 렌더링 ─────────────────────────────────────────────────────────────

def draw_overlay(frame, is_closed, closed_duration, alarm_active, confidence=0.0):
    """상태 오버레이를 프레임에 그립니다."""
    h, w = frame.shape[:2]

    # 알람 활성: 빨간 테두리
    if alarm_active:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

    # 눈 상태 텍스트
    if is_closed:
        label = f"CLOSED  {closed_duration:.1f}sec"
        color = (0, 0, 255)
    else:
        label = "OPEN"
        color = (0, 200, 0)

    cv2.putText(frame, label, (22, 52), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 2, cv2.LINE_AA)

    # closed 확률 바
    bar_x, bar_y, bar_w, bar_h = 20, 70, 200, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * confidence)
    bar_color = (0, 0, 220) if confidence >= CLOSED_CONFIDENCE else (0, 180, 0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"{confidence:.0%}", (bar_x + bar_w + 6, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 하단 안내
    guide = f"conf>{CLOSED_CONFIDENCE:.0%} | thr:{CLOSED_EYE_THRESHOLD}s | api | Q:quit"
    cv2.putText(frame, guide, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

    return frame


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  졸음 감지 프로그램 시작")
    print("=" * 55)

    # pygame mixer 초기화 (오디오)
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)

    # 알람 오디오 백그라운드 다운로드
    threading.Thread(target=_download_alarm_audio, args=(ALARM_YOUTUBE_URL,), daemon=True).start()

    # eye-status-api 연결 확인
    try:
        _check_api()
    except requests.RequestException as e:
        print(f"❌ eye-status-api에 연결할 수 없습니다 ({EYE_STATUS_API_URL}): {e}")
        sys.exit(1)

    # 추론 스레드 시작
    threading.Thread(target=_inference_worker, daemon=True).start()

    # 웹캠 초기화
    print("📷 웹캠 초기화 중...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✅ 준비 완료! 'q' 키로 종료\n")

    ensure_face_model()
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            num_faces=1,
        )
    )

    def _crop_eye(frame, landmarks, indices):
        h, w = frame.shape[:2]
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        x0 = max(0, min(p[0] for p in pts) - EYE_CROP_PAD)
        x1 = min(w, max(p[0] for p in pts) + EYE_CROP_PAD)
        y0 = max(0, min(p[1] for p in pts) - EYE_CROP_PAD)
        y1 = min(h, max(p[1] for p in pts) + EYE_CROP_PAD)
        crop = frame[y0:y1, x0:x1]
        return (crop, (x0, y0, x1, y1)) if crop.size > 0 else (None, None)

    # 상태 변수
    is_closed = False
    open_streak = 0      # 연속 "open" 판정 횟수 (OPEN_FORGIVENESS 도달 시 타이머 리셋)
    closed_start = None
    closed_duration = 0.0
    confidence = 0.0
    frame_count = 0
    eye_boxes = []
    alarm_window_open = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 웹캠 프레임 읽기 실패")
            break

        frame_count += 1

        try:
            # N 프레임마다 FaceLandmarker로 눈 크롭 후 추론 큐에 전송
            if frame_count % INFER_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = face_landmarker.detect(mp_img)
                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    left_crop,  left_box  = _crop_eye(frame, lm, LEFT_EYE_IDX)
                    right_crop, right_box = _crop_eye(frame, lm, RIGHT_EYE_IDX)
                    eye_boxes = [b for b in (left_box, right_box) if b is not None]
                    if left_crop is not None and right_crop is not None:
                        try:
                            _frame_q.put_nowait((left_crop.copy(), right_crop.copy()))
                        except queue.Full:
                            pass
                else:
                    eye_boxes = []

            # 눈 박스 시각화
            for (x0, y0, x1, y1) in eye_boxes:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 0), 1)

            # 최신 추론 결과 반영 (open_streak으로 일시적 오판 무시)
            try:
                result, confidence = _result_q.get_nowait()
                if result:
                    is_closed = True
                    open_streak = 0
                else:
                    open_streak += 1
                    if open_streak >= OPEN_FORGIVENESS:
                        is_closed = False
            except queue.Empty:
                pass

            # 눈 감은 시간 추적
            now = time.time()
            if is_closed:
                if closed_start is None:
                    closed_start = now
                closed_duration = now - closed_start
            else:
                closed_start = None
                closed_duration = 0.0

            # 알람 제어
            if closed_duration >= CLOSED_EYE_THRESHOLD and not _alarm_playing:
                threading.Thread(target=play_alarm, daemon=True).start()
            elif not is_closed and _alarm_playing:
                threading.Thread(target=stop_alarm, daemon=True).start()

            # 오버레이 + 화면 표시
            frame = draw_overlay(frame, is_closed, closed_duration, _alarm_playing, confidence)
            cv2.imshow("Drowsiness Detector", frame)

            # 알람 영상 프레임 표시 (메인 스레드에서만 cv2.imshow 호출)
            if _alarm_playing:
                try:
                    alarm_frame = _alarm_frame_q.get_nowait()
                    cv2.imshow(ALARM_WINDOW, alarm_frame)
                    alarm_window_open = True
                except queue.Empty:
                    pass
            elif alarm_window_open:
                try:
                    cv2.destroyWindow(ALARM_WINDOW)
                except Exception:
                    pass
                alarm_window_open = False
                # 알람 창이 닫히면 웹캠 창을 앞으로 가져옴
                try:
                    cv2.setWindowProperty("Drowsiness Detector", cv2.WND_PROP_TOPMOST, 1)
                    cv2.setWindowProperty("Drowsiness Detector", cv2.WND_PROP_TOPMOST, 0)
                except Exception:
                    pass

        except Exception as e:
            print(f"⚠️  루프 오류 (계속 실행): {e}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 정리
    _frame_q.put(None)
    face_landmarker.close()
    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("👋 프로그램 종료")


if __name__ == "__main__":
    main()
