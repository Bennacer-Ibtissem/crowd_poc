import streamlit as st

import cv2

import time

import numpy as np

from ultralytics import YOLO

import pandas as pd

from libs.evaluation import get_model_info, process_frame, calculate_performance_metrics

from libs.portrait_handler import get_video_orientation
 
st.set_page_config(page_title="Crowd in Mecca", layout="wide")
 
# Custom UI hiding

st.markdown("""
<style>

    [data-testid="stSidebarNav"], [data-testid="collapsedControl"] { display: none; }

    header, footer { visibility: hidden; }
</style>

""", unsafe_allow_html=True)
 
st.title("🕋 Crowd in Mecca ")
 
# Sidebar layout

st.sidebar.title("🎥 Video Input Mode")
 
video_input_mode = st.sidebar.radio(

    "Choose Input Source",

    options=["Upload", "Stream"],

    index=0,

    horizontal=True

)
 
uploaded_file = None

stream_url = ""
 
# Show file uploader only in Upload mode

if video_input_mode == "Upload":

    uploaded_file = st.sidebar.file_uploader(

        "📁 Browse a video file", type=["mp4", "avi", "mov"]

    )
 
# Show RTMP input only in Stream mode

if video_input_mode == "Stream":

    stream_url = st.sidebar.text_input(

        "🔗 Stream URL", 

        value="rtmp://10.30.1.181:8080/live/stream", 

        help="Enter your stream address."

    )
 
# Shared settings

st.sidebar.title("🔧 Model Settings")

available_models = [

    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"

]

selected_model = st.sidebar.selectbox("Select Model", available_models)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
 
# Portrait options only for upload

is_portrait = False

force_rotate = False

if video_input_mode == "Upload":

    st.sidebar.subheader("📱 Mobile Video Options")

    is_portrait = st.sidebar.checkbox("Portrait Video", value=False)

    force_rotate = st.sidebar.checkbox("Auto-rotate to Landscape", value=False)
 
# Load model with caching

@st.cache_resource

def load_model(path):

    return YOLO(path)
 
# --- Upload Mode ---

if video_input_mode == "Upload":

    if uploaded_file is not None:

        col1, col2 = st.columns(2)

        input_video_path = "temp_video.mp4"

        with open(input_video_path, "wb") as f:

            f.write(uploaded_file.getbuffer())
 
        with col1:

            st.subheader("📊 Original Video")

            cap = cv2.VideoCapture(input_video_path)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps = int(cap.get(cv2.CAP_PROP_FPS))
 
            if not is_portrait:

                ret, frame = cap.read()

                if ret:

                    auto_orientation = get_video_orientation(frame)

                    is_portrait = auto_orientation == "portrait"

                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            cap.release()
 
            orientation_text = "Portrait (Mobile)" if is_portrait else "Landscape"

            st.info(f"Detected Video Orientation: {orientation_text}")

            st.video(input_video_path)
 
        if st.button("🚀 Start Processing"):

            with st.spinner(f"Loading model {selected_model}..."):

                model = load_model(selected_model)

                model_size, num_params = get_model_info(selected_model)
 
            cap = cv2.VideoCapture(input_video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps = int(cap.get(cv2.CAP_PROP_FPS))
 
            with col2:
                st.subheader("🎬 Live Processing")
                if is_portrait:
                    st.info("📱 Processing Portrait Video" + (" (Auto-rotating)" if force_rotate else ""))
                else:
                    st.info("🎥 Processing Video")
                frame_placeholder = st.empty()

            progress_text = st.empty()

            metrics_placeholder = st.empty()

            frame_count = 0

            total_inference_time = 0

            total_objects_detected = 0

            fps_list = []

            object_counts = []
 
            while cap.isOpened():

                success, frame = cap.read()

                if not success:

                    break
 
                processed_frame, count, inference_time = process_frame(

                    frame, model, conf_threshold, [0],

                    is_portrait=is_portrait, force_rotate=force_rotate

                )
 
                frame_count += 1

                total_inference_time += inference_time

                total_objects_detected += count

                fps_list.append(1.0 / inference_time if inference_time > 0 else 0)

                object_counts.append(count)
 
                if frame_count % 5 == 0:

                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                    frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
 
                    progress_percentage = (frame_count / total_frames) * 100

                    progress_text.markdown(f"### Processing: {frame_count}/{total_frames} frames ({progress_percentage:.2f}%)")
 
                    metrics_data = calculate_performance_metrics(

                        video_length=total_frames / fps,

                        frame_size=(w, h),

                        model_name=selected_model,

                        num_params=num_params,

                        avg_fps=sum(fps_list[-20:]) / len(fps_list[-20:]),

                        avg_inference_time=total_inference_time / frame_count,

                        total_inference_time=total_inference_time,

                        total_objects_detected=total_objects_detected,

                        video_name=uploaded_file.name,

                        total_frames=total_frames,

                        avg_objects_per_frame=total_objects_detected / frame_count,

                        max_objects_detected=max(object_counts),

                        is_portrait=is_portrait,

                    )

                    metrics_placeholder.dataframe(pd.DataFrame(metrics_data), hide_index=True)
 
            cap.release()

            st.success(f"✅ Done! Processed {frame_count} frames.")
 
# --- Stream Mode ---

elif video_input_mode == "Stream":

    st.info(f"🔴 Connecting to stream: `{stream_url}`")

    cap = cv2.VideoCapture(stream_url)
 
    if not cap.isOpened():

        st.error("❌ Could not open stream. Check the URL or network.")

    else:

        with st.spinner(f"Loading model {selected_model}..."):

            model = load_model(selected_model)

            model_size, num_params = get_model_info(selected_model)
 
        frame_placeholder = st.empty()

        fps_display = st.empty()

        object_display = st.empty()
 
        st.success("✅ Streaming started. Press Stop or refresh to exit.")

        stop_button = st.button("🛑 Stop Stream")
 
        while cap.isOpened() and not stop_button:

            ret, frame = cap.read()

            if not ret:

                st.warning("⚠️ Stream ended or cannot read frame.")

                break
 
            processed_frame, count, inference_time = process_frame(

                frame, model, conf_threshold, [0],

                is_portrait=False, force_rotate=False

            )
 
            display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
 
            fps_display.markdown(f"**FPS:** {1.0/inference_time:.2f}")

            object_display.markdown(f"**People Detected:** {count}")
 
            time.sleep(0.02)
 
        cap.release()

 