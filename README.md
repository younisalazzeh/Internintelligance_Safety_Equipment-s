# YOLOv11 Object Detection

This project trains and evaluates a YOLOv11 model for object detection using a custom dataset from Roboflow.

## 📦 Dataset Download
We use Roboflow to fetch and prepare the dataset.
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("apd-0zj3f").project("innowork-24")
version = project.version(4)
dataset = version.download("yolov11")
```

## 🏋️ Training the Model
We use the `ultralytics` YOLO library to train the model.
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # Load pretrained model
results = model.train(data="/content/Innowork-24-4/data.yaml", epochs=50, imgsz=640, device=0)
```

## 📊 Model Evaluation
After training, we validate the model on test data.
```python
results = model.val(data="/content/Innowork-24-4/data.yaml")
print(results)
```

## 🔍 Performance Metrics
We calculate precision and recall using ground truth bounding boxes.
```python
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_true, y_pred) if y_true else 0
recall = recall_score(y_true, y_pred) if y_true else 0
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

## 📌 How to Run
1. Run the training script:
   ```sh
   python file.py
   ```


## 📸 Sample Output
(Screenshot of detection results)

## 🤖 Authors
- Yousef Ahmad Alazzeh
