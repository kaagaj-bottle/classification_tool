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


# model_path = r"runs\classify\train3\weights\best.onnx"
# model = cv2.dnn.readNetFromONNX(model_path)
parent_folder = "ss/val"
IMG_HEIGHT, IMG_WIDTH = 215, 172
N_CHANNEL = 3
N_BOX = 20
BOX_HEIGHT, BOX_WIDTH = 34, 34
pprint("Loading the model..")
model_yolo_path = r"runs/classify/train3/weights/best.pt"
model_yolo = YOLO(model_yolo_path, verbose=False)
pprint("Model Loaded...")

total_time_for_single_batches = 0


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        global total_time_for_single_batches
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time_for_single_batches += (end_time-start_time)
        return result

    return wrapper


def get_backpack_imgs(parent_folder: str) -> Tuple[np.array, List[str]]:
    backpack_list = glob.glob(f"{parent_folder}/*.png")
    len_backpack_list = len(backpack_list)
    image_array = np.zeros(
        (len_backpack_list, IMG_HEIGHT, IMG_WIDTH, N_CHANNEL), dtype=np.uint8)
    for idx in range(len_backpack_list):
        image_array[idx] = cv2.imread(backpack_list[idx])

    return image_array, backpack_list


def get_division_rects():
    rects = []
    cur_x, cur_y = 10, 18
    bx_h, bx_w = BOX_HEIGHT, BOX_WIDTH
    padd = 3
    while cur_y + bx_h <= IMG_HEIGHT:
        cur_x = 10
        while cur_x + bx_w < IMG_WIDTH:
            rect = (cur_x, cur_y, bx_w, bx_h)
            rects.append(rect)
            cur_x = cur_x + padd + bx_w
        cur_y = cur_y + padd + bx_h
    return rects


@timing_decorator
def divide_backpack_to_items(image_array: np.array) -> np.array:
    # the actual size of item image is 32x32, each side (border) is of size 1 pixel
    divided_images_array = np.zeros(
        (image_array.shape[0], N_BOX, BOX_HEIGHT-2, BOX_WIDTH-2, N_CHANNEL), dtype=np.uint8)
    coordinates = get_division_rects()
    for idx in range(divided_images_array.shape[0]):
        division_counter = 0
        for x, y, w, h in coordinates:
            divided_image = image_array[idx][y+1:y+h-1, x+1:x+w-1]
            # divided_images_array[idx][division_counter] = np.transpose(
            #     divided_image, (2, 0, 1))
            divided_images_array[idx][division_counter] = divided_image
            division_counter += 1

    return divided_images_array


def classify_divided_image_opencv(divided_image_array: np.array):
    blobs = cv2.dnn.blobFromImages(
        divided_image_array, 1, (BOX_WIDTH-2, BOX_HEIGHT-2), swapRB=True, crop=False).transpose(0, 3, 1, 2)
    results = []
    try:
        for idx, blob in enumerate(blobs):

            model.setInput(blob)
            output = model.forward()
            results.append(output)
    except Exception as e:
        print(e)
        return None

    return results
    print(divided_image_array.shape)
    output = model_yolo(divided_image_array, verbose=False)
    return output


@timing_decorator
def classify_divided_image_ultralytics(divided_image_tensor: torch.Tensor) -> List[List[ultralytics.engine.results.Results]]:
    outputs = model_yolo(divided_image_tensor, verbose=False)
    return outputs


@timing_decorator
def get_final_result(divided_images_array: torch.Tensor) -> List[List[List[ultralytics.engine.results.Results]]]:
    outputs: List[List[List[ultralytics.engine.results.Results]]] = []
    pprint("Running Inference...")
    for idx in tqdm(range(divided_images_array.shape[0])):
        cur_output = classify_divided_image_ultralytics(
            divided_images_array[idx])
        outputs.append(cur_output)
    return outputs


def format_output(outputs: List[List[List[ultralytics.engine.results.Results]]], backpack_list: List[str]) -> Dict:
    output_dict: Dict = {}
    for idx, img in enumerate(backpack_list):
        output_dict[img] = [item.names[item.probs.top1]
                            for item in outputs[idx]]
    return output_dict


def write_output_to_csv(formatted_output: Dict, out_file: str) -> None:
    with open(out_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in formatted_output.items():
            writer.writerow([key, value])


if __name__ == '__main__':
    init_time = time.time()
    image_array, backpack_list = get_backpack_imgs(parent_folder)
    divided_images_array = np.transpose(divide_backpack_to_items(
        image_array), (0, 1, 4, 2, 3)).astype(np.float32)/255.0
    divided_images_tensor = torch.from_numpy(divided_images_array)
    outputs: List[List[ultralytics.engine.results.Results]] = get_final_result(
        divided_images_tensor)
    print("Writing output to file...")
    formatted_output = format_output(outputs, backpack_list)
    write_output_to_csv(formatted_output, "output/o1.csv")
    print(f"Avg time for single batch: {total_time_for_single_batches/201}")
