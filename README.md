# 🚦 Smart Traffic Detection System using YOLOv8 + Intel OpenVINO 🚗

---

## 🎯 Project Overview

This project demonstrates real-time **Traffic Object Detection** using:

- **YOLOv8 (You Only Look Once - Version 8)** deep learning model for object detection.
- **Intel OpenVINO Toolkit** for efficient inference and hardware-accelerated deployment.
- **OpenCV** for video processing and frame rendering.
- **Google Colab** for model export, preprocessing & cloud experimentation.

---

## 💡 Why This Project?

Traffic surveillance is a critical AI problem:
- Helps monitor traffic flow 🚦
- Detects vehicles 🚌 🚗 🏍 in real-time
- Can be used for smart cities, congestion detection, traffic violations, and much more.

By combining **YOLOv8's accuracy** with **OpenVINO's inference speed**, this project provides a scalable, efficient solution for edge deployment as well as cloud deployment.

---

## 🔬 Technology Stack

| Technology | Usage |
| ----------- | ----------- |
| **YOLOv8 (Ultralytics)** | State-of-the-art object detection model |
| **Intel OpenVINO Toolkit** | Optimized deep learning inference engine |
| **OpenCV** | Real-time video processing |
| **Python 3.10+** | Primary programming language |
| **Google Colab** | Model conversion and experimentation platform |
| **PyTorch Backend** | Original YOLOv8 model trained on COCO dataset |

---

## 🚀 Full Workflow

### 1️⃣ Model Exporting (Google Colab)

- YOLOv8 model (`yolov8n.pt`) was downloaded and exported to OpenVINO format using Ultralytics library.
- Google Colab was used to run the following code:

```python
from ultralytics import YOLO

# Load YOLOv8n pretrained model
model = YOLO("yolov8n.pt")

# Export model to OpenVINO IR format
model.export(format='openvino')
````

* This generates files like:

```
yolov8n_openvino_model/
 ├── openvino_model.xml
 └── openvino_model.bin
```

### 2️⃣ Project Structure (Local VS Code)

```bash
traffic_detection_ai_openvino/
│
├── models/
│   └── yolov8n_openvino_model/
│         ├── openvino_model.xml
│         └── openvino_model.bin
│
├── video/
│   └── traffic_sample.mp4
│
├── main.py          <-- Main detection code
├── notebook.ipynb   <-- Colab notebook version (model export)
├── requirements.txt <-- Python dependencies
└── README.md        <-- This documentation
```

### 3️⃣ Running the Project

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run Real-Time Detection

```bash
python main.py
```

* The model reads frames from `video/traffic_sample.mp4`.
* Performs real-time inference using OpenVINO.
* Draws bounding boxes for detected vehicles.
* Displays live video with detections.

### 4️⃣ Output Example

✅ Detected objects:

* Car 🚗
* Bus 🚌
* Truck 🚚
* Motorcycle 🏍️
* Person 🚶 (if present)

✅ Each detection shows:

* Bounding Box
* Label
* Confidence Score

✅ Inference is fully hardware-accelerated using **Intel OpenVINO**.

---

## ⚙ Model Used

* Model: `YOLOv8n` (Nano model — fast and light)
* Dataset: MS COCO 80 classes
* Format: OpenVINO IR

👉 You can easily swap this with larger YOLOv8 models (`yolov8s`, `yolov8m`, etc) for higher accuracy.

---

## 🌐 Why OpenVINO?

| Benefit              | Why?                                |
| -------------------- | ----------------------------------- |
| 💨 Inference Speed   | Highly optimized for CPU, GPU, VPU  |
| ⚡ Lightweight        | Smaller models, less resource usage |
| 🖥 Hardware Friendly | Supports multiple Intel devices     |
| 📦 Industry Ready    | Used in production edge deployments |

---

## 🎯 Future Extensions

* Deploy on live camera streams (instead of video file)
* Edge AI deployment on Raspberry Pi / Intel NUC
* Deploy as full web dashboard or cloud API
* Train YOLOv8 on custom dataset for specific traffic scenarios
* Add vehicle counting, speed estimation, violation detection

---

## 📂 Dependencies

All dependencies are listed in `requirements.txt`:

```bash
ultralytics==8.0.151
openvino==2024.0.0
opencv-python
numpy
```

---

## 🧪 Tested Environment

* Windows 11 + VS Code
* Python 3.10
* OpenVINO 2024.0
* Ultralytics YOLOv8 8.0.151
* Google Colab for initial model export

---

## 🔗 Resources

* [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
* [Intel OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
* [OpenCV](https://opencv.org/)
* [Google Colab](https://colab.research.google.com/)

---

## 🙏 Acknowledgement

This project combines the best of open-source & Intel optimizations to build scalable traffic detection systems for smart cities and research.

---

## 🎯 Summary

✅ **Full end-to-end AI pipeline**
✅ **YOLOv8 + OpenVINO hybrid architecture**
✅ **Real-time object detection for traffic videos**
✅ **Google Colab + VS Code cross-platform setup**
✅ **Perfect starting point for Edge AI + Smart City applications**

---

> 🔥 *If you like this repo, feel free to ⭐ star it — contributions & forks are always welcome!*

```

