import streamlit as st
import cv2
import os
from ultralytics import solutions
import ultralytics
import time

# Function to count specific classes in a video
def count_specific_classes(video_path, output_video_path, model_path, classes_to_count):
    """Count specific classes of objects in a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_points = [(20, 1), (w, 1)]    
    counter = solutions.ObjectCounter(show=True, region=line_points, model=model_path, classes=classes_to_count)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = counter(im0)
        num_objects = results.total_tracks

        cv2.putText(im0, f"Tracked number of person: {num_objects}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

# Streamlit interface
st.title("YOLO Video Detection")
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    video_path = uploaded_file.name
    with open(video_path, mode='wb') as f:
        f.write(uploaded_file.getbuffer())
    st.video(video_path)

    output_video_path = "output_specific_classes.mp4"
    count_specific_classes(video_path, output_video_path, "yolo11n.pt", [0,41,39])
    st.video(output_video_path)
