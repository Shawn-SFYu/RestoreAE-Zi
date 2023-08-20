"""
organize imgs from AUTHOR -> denoise, upsample, scale to 512, save to pickle files
"""
import os
import logging
import cv2
from GenBaseImg import ChartoImg
import numpy as np
import pickle



class DataPair:
    def __init__(self, img_input, img_output):
        self.input = img_input
        self.output = img_output

def ImgPrep(path: str):
    img = cv2.imread(path)
    denoised = cv2.bilateralFilter(img, 13, 100, 100)
    img_hr = esdr.upsample(denoised)

    """Basic parameters after super-reso"""
    height, width = img_hr.shape[:2]
    center = np.array([width // 2, height // 2])

    """Thresholding"""
    # this may not be desired due to the jigsaw effect
    gray_img = cv2.cvtColor(img_hr, cv2.COLOR_RGB2GRAY)
    thresh_value, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    """Img mode detection light/dark"""
    dark_distance = 0
    light_distance = 0
    dark_count = 0
    light_count = 0
    for x in range(width):
        for y in range(height):
            if thresh_img[y, x] == 0:
                dark_distance += np.linalg.norm(center - np.array([y, x]))
                dark_count += 1
            else:
                light_distance += np.linalg.norm(center - np.array([y, x]))
                light_count += 1
    dark_distance = dark_distance / dark_count
    light_distance = light_distance / light_count
    light_mode = True if light_distance < dark_distance else False
    # print(f"Light mode is {light_mode}")
    if light_mode == False:
        thresh_img = cv2.bitwise_not(thresh_img)
        gray_img = cv2.bitwise_not(gray_img)
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(thresh_img, kernel, iterations=1)

    """Scale to dimension"""
    masked_img = cv2.bitwise_and(gray_img, gray_img, mask=dilated_mask)
    desired_resolution = 512

    max_dim = max(height, width)
    square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
    x_offset = (max_dim - width) // 2
    y_offset = (max_dim - height) // 2
    square_img[y_offset:y_offset + height, x_offset:x_offset + width] = masked_img
    resized_img = cv2.resize(square_img, (desired_resolution, desired_resolution), \
                             interpolation=cv2.INTER_LINEAR)
    return resized_img

if __name__ == "__main__":
    AUTHOR = '赵孟頫'
    DIR = os.getcwd()
    
    model_path = "EDSR_x4.pb"
    esdr = cv2.dnn_superres.DnnSuperResImpl_create()
    esdr.readModel(model_path)
    esdr.setModel("edsr", 4)

    with open('characters.txt', 'r') as file:
        # Read the contents of the file into a single string
        contents = file.read()

    os.chdir(os.path.join(DIR, AUTHOR))
    DIR = os.getcwd()
    folders = os.listdir(DIR)

    logging.basicConfig(filename='DataPrep.log', level=logging.DEBUG)

    img_dict = {}
    for i, character in enumerate(contents):
        logging.debug(f"Processing {i} {character} \n")
        assert character in folders
        clips = os.listdir(os.path.join(DIR, character))
        if len(clips) != 0:
            if character not in img_dict:
                img_dict[i] = []
                logging.debug(f"entry created \n")
            for path in clips:
                img_input = ChartoImg(character)
                img_output = ImgPrep(os.path.join(DIR, character, path))
                img_dict[i].append(DataPair(img_input, img_output))
        div_n, mod_n = divmod(i+1, 400)
        logging.debug(f" At {div_n} {mod_n} \n")
        if mod_n == 0:
            output_name = "img_dict_" + str(div_n) + AUTHOR + ".pickle"
            with open(output_name, "wb") as f:
                pickle.dump(img_dict, f)
                img_dict = {}
    with open("img_dict_0" + AUTHOR + ".pickle", "wb") as f:
        pickle.dump(img_dict, f)
