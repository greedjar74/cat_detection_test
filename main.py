import cv2
import av
import time
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('/Users/kimhongseok/cat_detection_test/best.pt')

# Streamlit UI 설정
st.title("📸 실시간 고양이 탐지 - YOLO + Streamlit")
st.write("실시간 웹캠 영상을 통해 고양이를 탐지합니다.")

# 비디오 처리 클래스 정의
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_time = time.time()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # 객체 탐지
        results = model.predict(image, conf=0.1)

        # 결과 처리
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf >= 0.2:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f'{conf:.2f}', (x1, y1 - 10), self.font, 0.8, (0, 255, 0), 2)

        # FPS 계산
        cur_time = time.time()
        sec = cur_time - self.prev_time
        self.prev_time = cur_time
        fps = 1 / sec if sec > 0 else 0
        cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), self.font, 1, (0, 255, 0), 2)

        return image

# WebRTC 스트리밍 실행
webrtc_streamer(
    key="object-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
