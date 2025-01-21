import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from typing import List
from pathlib import Path
import glob


model_path = r"runs\classify\train3\weights\best.onnx"
model = cv2.dnn.readNetFromONNX(model_path)
parent_folder = "ss/val"
IMG_HEIGHT, IMG_WIDTH = 215, 172
N_CHANNEL = 3
DIVISIONS = 20
DIVISION_HEIGHT, DIVISION_WIDTH = 32, 32

model_yolo_path = r"runs\classify\train3\weights\best.pt"
model_yolo = YOLO(model_yolo_path)


def get_backpack_imgs(parent_folder: str) -> List[str]:
    backpack_list = glob.glob(f"{parent_folder}/*.png")
    len_backpack_list = len(backpack_list)
    image_array = np.zeros(
        (len_backpack_list, IMG_HEIGHT, IMG_WIDTH, N_CHANNEL), dtype=np.uint8)
    for idx in range(len_backpack_list):
        image_array[idx] = cv2.imread(backpack_list[idx])

    return image_array


def get_division_rects():
    rects = []
    cur_x, cur_y = 10, 18
    bx_w, bx_h = DIVISION_HEIGHT, DIVISION_WIDTH
    padd = 3
    while cur_y + bx_h <= IMG_HEIGHT:
        cur_x = 10
        while cur_x + bx_w < IMG_WIDTH:
            rect = (cur_x, cur_y, bx_w, bx_h)
            rects.append(rect)
            cur_x = cur_x + padd + bx_w
        cur_y = cur_y + padd + bx_h
    return rects


def divide_backpack_to_items(image_array: np.array) -> np.array:

    divided_images_array = np.zeros(
        (image_array.shape[0], DIVISIONS, DIVISION_HEIGHT, DIVISION_WIDTH, N_CHANNEL), dtype=np.uint8)
    coordinates = get_division_rects()
    for idx in range(divided_images_array.shape[0]):
        division_counter = 0
        for x, y, w, h in coordinates:
            divided_image = image_array[idx][y:y+h, x:x+w]
            # divided_images_array[idx][division_counter] = np.transpose(
            #     divided_image, (2, 0, 1))
            divided_images_array[idx][division_counter] = divided_image
            division_counter += 1

    return divided_images_array


def classify_divided_image(divided_image_array: np.array):

    # blob = cv2.dnn.blobFromImages(
    #     divided_image_array, 1.0, (DIVISION_WIDTH, DIVISION_HEIGHT), swapRB=True, crop=False)
    # try:

    #     model.setInput(divided_image_array)
    #     output = model.forward()
    #     output
    # except Exception as e:
    #     print(e)
    #     return None
    print(divided_image_array.shape)
    output = model_yolo(divided_image_array, verbose=False)
    return output


def get_final_result(divided_images_array: np.array):
    outputs = []
    for idx in range(divided_images_array.shape[0]):
        cur_output = classify_divided_image(divided_images_array[idx])
        outputs.append(cur_output)
        break
    print(outputs)
    return outputs


def show_outputs(outputs):
    for output in outputs:
        output.show()
        output.save(filename="output.jpg")


if __name__ == '__main__':
    image_array = get_backpack_imgs(parent_folder)
    divided_images_array = divide_backpack_to_items(image_array)
    outputs = get_final_result(divided_images_array=divided_images_array)

    print(outputs)
