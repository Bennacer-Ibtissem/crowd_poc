import streamlit as st

import cv2

import os

import time

from ultralytics import solutions

import warnings

# Suppress warnings

warnings.filterwarnings('ignore')

# Function to count specific classes in the video and show detection counts

def count_and_display_classes(video_path, model_path, classes_to_count):

    """Count specific classes of objects in a video and show it live in Streamlit."""

    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened(), "Error reading video file"

    # Get video properties

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Output video path to save processed video

    output_video_path = "output_live_processed.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Initialize ObjectCounter

    line_points = [(20, 1), (w, 1)]

    counter = solutions.ObjectCounter(show=True, region=line_points, model=model_path, classes=classes_to_count)

    # Create Streamlit frame placeholder

    frame_placeholder = st.empty()

    frame_count = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            break

        # Process frame to detect objects

        results = counter(frame)

        # Extract number of detected objects

        num_objects = results.total_tracks

        # Add text to the frame with the detected count

        cv2.putText(frame, f"Detected objects: {num_objects}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        # Optional: Save processed frame to output video

        out.write(results.plot_im)

        # Convert to RGB for Streamlit display

        processed_frame_rgb = cv2.cvtColor(results.plot_im, cv2.COLOR_BGR2RGB)

        # Display the frame live in Streamlit

        frame_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)

        frame_count += 1

        # Optional: slow down processing to match original FPS

        time.sleep(1 / fps)

    cap.release()

    out.release()

    cv2.destroyAllWindows()

    st.success(f"Processing completed! Processed {frame_count} frames.")

    return output_video_path

# Streamlit UI Setup

st.title("Crowd detection and counting")

# Upload video

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    # Save uploaded file temporarily

    input_video_path = "uploaded_video.mp4"

    with open(input_video_path, 'wb') as f:

        f.write(uploaded_file.getbuffer())

    # Show the original uploaded video

    st.subheader("Original Video")

    st.video(input_video_path)

    # Settings for model and classes to count

    st.sidebar.title("Settings")

    model_path = st.sidebar.text_input("Model Path", "yolo11n.pt")

    classes_to_count = st.sidebar.multiselect("Classes to Count", options=[0, 39, 41], default=[0])

    # Start processing the video

    if st.button("Start Processing"):

        with st.spinner('Processing video...'):

            output_path = count_and_display_classes(input_video_path, model_path, classes_to_count)

        # Display the processed video

        st.subheader("Processed Video (Saved Output)")

        with open(output_path, 'rb') as f:

            st.video(f.read())
