import cv2
stream_url = "rtmp://10.30.1.181:443/live/stream"
cap = cv2.VideoCapture(stream_url)
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Process the frame here
    cv2.imshow("Live Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()