import cv2
import av
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# YOLO 모델 로드 (GPU 사용 가능 시 아래 주석 해제)
# model = YOLO('best.pt').half().to('cuda')
model = YOLO('best.pt')

# Streamlit 페이지 구성
st.title("🐱 실시간 고양이 탐지 (YOLO + Streamlit)")
st.markdown("웹캠을 통해 고양이를 탐지하고, 가장 높은 확률을 점수로 표시합니다.")

# 점수 출력용 공간
score_placeholder = st.empty()

# 비디오 처리 클래스
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_time = time.time()
        self.frame_skip = 2  # 2프레임마다 1프레임 처리
        self.frame_count = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def transform(self, frame):
        self.frame_count += 1

        # 프레임 건너뛰기 (속도 향상용)
        if self.frame_count % self.frame_skip != 0:
            return frame.to_ndarray(format="bgr24")

        image = frame.to_ndarray(format="bgr24")

        # 프레임 자체 해상도 축소 (속도 향상)
        image = cv2.resize(image, (640, 360))

        # YOLO 예측 (입력 해상도 축소)
        results = model.predict(image, conf=0.5, imgsz=320, verbose=False)

        max_conf = 0  # 최고 확률 저장

        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf >= 0.2:
                    if conf > max_conf:
                        max_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f'{conf:.2f}', (x1, y1 - 10), self.font, 0.8, (0, 255, 0), 2)

        # FPS 계산 및 출력
        cur_time = time.time()
        elapsed = cur_time - self.prev_time
        self.prev_time = cur_time
        fps = 1 / elapsed if elapsed > 0 else 0
        cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), self.font, 1, (0, 255, 0), 2)

        # Streamlit 점수 표시 (웹캠 아래 큰 글씨)
        score_placeholder.markdown(
            f"<h1 style='text-align: center; color: red;'>Score: {max_conf:.2f}</h1>",
            unsafe_allow_html=True
        )

        return image

# WebRTC 웹캠 실행
webrtc_streamer(
    key="cat-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
