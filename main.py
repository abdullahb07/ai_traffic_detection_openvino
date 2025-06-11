import cv2
import numpy as np
from openvino.runtime import Core

# Initialize OpenVINO runtime
ie = Core()
model_ir = ie.read_model(model="models/yolov8n_openvino_model/openvino_model.xml")
compiled_model = ie.compile_model(model=model_ir, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Load your video
cap = cv2.VideoCapture("video/traffic_sample.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing for OpenVINO
    resized = cv2.resize(frame, (640, 640))
    input_image = resized.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32) / 255.0

    # Inference
    result = compiled_model([input_image])[output_layer]

    # For now, we just print running inference frame by frame
    print("Frame processed")

cap.release()
cv2.destroyAllWindows()
