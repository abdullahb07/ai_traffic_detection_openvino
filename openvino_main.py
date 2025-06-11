import cv2
import numpy as np
from openvino.runtime import Core

# Initialize OpenVINO runtime
ie = Core()

# Load OpenVINO exported YOLOv8n model
model_ir = ie.read_model(model="models/yolov8n_openvino_model/openvino_model.xml")
compiled_model = ie.compile_model(model=model_ir, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Class names from COCO dataset (YOLO default classes)
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Open video
cap = cv2.VideoCapture("video/traffic_sample.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    resized = cv2.resize(frame, (640, 640))
    input_image = resized.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32) / 255.0

    # Inference
    result = compiled_model([input_image])[output_layer]

    # Post-processing
    result = np.squeeze(result)  # (batch, boxes, 85)
    boxes = result[:, :4]
    scores = result[:, 4]
    class_probs = result[:, 5:]
    class_ids = np.argmax(class_probs, axis=1)
    confidences = scores * class_probs[np.arange(len(class_probs)), class_ids]

    # Thresholds
    conf_threshold = 0.4
    iou_threshold = 0.5

    # Filter boxes
    indices = np.where(confidences > conf_threshold)[0]

    # Rescale boxes back to original frame size
    for i in indices:
        x, y, w, h = boxes[i]
        x1 = int((x - w / 2) * frame.shape[1] / 640)
        y1 = int((y - h / 2) * frame.shape[0] / 640)
        x2 = int((x + w / 2) * frame.shape[1] / 640)
        y2 = int((y + h / 2) * frame.shape[0] / 640)

        label = class_names[class_ids[i]]
        confidence = confidences[i]

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
