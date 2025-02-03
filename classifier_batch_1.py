from sklearn.metrics import precision_score, recall_score, f1_score
import ast
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from typing import List, Tuple, Dict
from pathlib import Path
import glob
from pprint import pprint
import csv
import torch
import ultralytics.engine
import ultralytics.engine.results
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
import onnx

print("Loading the onnx model")
model_path = r"runs_batch_1/classify/train3/weights/best.onnx"
model = cv2.dnn.readNetFromONNX(model_path)
print("ONNX model loaded")
IMG_HEIGHT, IMG_WIDTH = 215, 172
N_CHANNEL = 3
N_BOX = 20
BOX_HEIGHT, BOX_WIDTH = 34, 34
model_yolo = YOLO(model="runs_batch_1/classify/train3/weights/best.pt")
model_transformations = model_yolo.transforms

MODE_NO_TRANSFORM = 0
MODE_CUSTOM_TRANSFORM = 1
MODE_MODEL_TRANSFORM = 2
MODEL_MODE_ULTRALYTICS = 0
MODEL_MODE_ONNX = 1

model_transform = torchvision.transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(
        32, 32), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(size=(32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
])

onnx_model = onnx.load(model_path)
names_dict = onnx_model.metadata_props[10].value
labels_dict = ast.literal_eval(names_dict)


def resize(image, size=32):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def convertToFloat(image):
    return image.astype(np.float32)/255.0


def center_crop(image, crop_size=(32, 32)):
    h, w, _ = image.shape
    ch, cw = crop_size
    start_x = (w - cw) // 2
    start_y = (h - ch) // 2
    return image[start_y:start_y+ch, start_x:start_x+cw]


def to_tensor(image):
    return image.astype(np.float32) / 255.0  # Convert to [0, 1] range


def normalize(image, mean=np.array([0.0, 0.0, 0.0]), std=np.array([1.0, 1.0, 1.0])):
    return (image - mean) / std


def custom_transform(image):
    image = resize(image, size=32)
    image = center_crop(image)
    image = convertToFloat(image)
    image = image.transpose(2, 0, 1)

    return image


def transform_using_torch_compose(img: np.array):
    img = model_transform(img).numpy()
    return img


def classify_divided_image_ultralytics(divided_image_tensor: torch.Tensor) -> List[List[ultralytics.engine.results.Results]]:
    outputs = model_yolo(divided_image_tensor, verbose=False)
    return outputs


def classify_divided_image(divided_image: np.array, model_mode, mode):
    if model_mode == MODEL_MODE_ONNX:
        transformed_divided_image = np.zeros((1, 3, 32, 32), dtype=np.float32)
        if mode == MODE_CUSTOM_TRANSFORM:
            transformed_divided_image[0] = custom_transform(divided_image)
        elif mode == MODE_MODEL_TRANSFORM:
            transformed_divided_image[0] = transform_using_torch_compose(
                divided_image)
        else:
            transformed_divided_image[0] = np.transpose(
                divided_image, (0, 3, 1, 2))
        model.setInput(transformed_divided_image)
        output = model.forward()
        return output
    elif model_mode == MODEL_MODE_ULTRALYTICS:
        output = model_yolo(divided_image, verbose=False)
        return output


def get_final_result(divided_images_array, model_mode=MODEL_MODE_ULTRALYTICS, mode=MODE_NO_TRANSFORM):
    outputs = []
    pprint("Running Inference...")
    for idx in tqdm(range(divided_images_array.shape[0])):
        cur_output = classify_divided_image(
            divided_images_array[idx], model_mode, mode)
        outputs.append(cur_output)
    return outputs, model_mode


parent_dir = "data/train"
classes = [f"{item}" for item in os.listdir(parent_dir)]
mapping = {}
for item in classes:
    class_png_files = glob.glob(f"{parent_dir}/{item}/*.png")
    [mapping.update({file: item}) for file in class_png_files]


def get_image_array(file_list, model_mode):
    image_array = np.zeros((len(file_list), 32, 32, 3), dtype=np.uint8)
    for idx, file in enumerate(file_list):
        img = cv2.imread(file)
        if img.shape == (32, 32, 3):
            if model_mode == MODEL_MODE_ONNX:
                image_array[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif model_mode == MODEL_MODE_ULTRALYTICS:
                image_array[idx] = img
        else:
            continue
    return image_array


image_array = get_image_array(list(mapping.keys()), model_mode=MODEL_MODE_ONNX)

predictions, model_mode = get_final_result(
    image_array, model_mode=MODEL_MODE_ONNX, mode=MODE_CUSTOM_TRANSFORM)


def get_prediction_texts(predictions, model_mode):
    predictions_text = []
    if model_mode == MODEL_MODE_ONNX:
        predictions_text = [
            labels_dict[int(np.argmax(probs))] for probs in predictions]
    elif model_mode == MODEL_MODE_ULTRALYTICS:

        predictions_text = [output[0].names[output[0].probs.top1]
                            for output in predictions]
    return predictions_text


predictions_text = get_prediction_texts(predictions, model_mode)

true_values = list(mapping.values())

precision = precision_score(
    true_values, predictions_text, average="weighted", zero_division=np.nan)
recall = recall_score(true_values, predictions_text,
                      average="weighted", zero_division=np.nan)
print("Precision: ", precision, "Recall", recall)
