# type:ignore
import torch
from pathlib import Path
import time
import cv2
import ultralytics
from ultralytics import YOLO
from constant import DATA_IAMGES_PATH

device: str = "cuda" torch.cuda.is_available() else "cpu"
# Load a model
# this is pretrained model loaded from ultralytics
model = YOLO("yolo11n-cls")

# Train the model
train_results = model.train(
    data=str(DATA_IAMGES_PATH),  # path to dataset YAML
    # each image is 32x32 for classification
    imgsz=32,
    epochs=300,
    device=device,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
pred = list(Path(r"datasets\images\train\shovel").glob("*"))[-1]
results = model(pred)
results[0].save()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
