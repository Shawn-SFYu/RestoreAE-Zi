"""
Prepare painted data from DataPair into image folders
"""

import os
import logging
import glob
from gen_base_img import char2img
from PIL import Image, ImageFont


if __name__ == "__main__":
    data_path = os.getcwd()
    data_path = os.path.join(data_path, "DicData")
    fonts = glob.glob("Fonts/*.ttf")
    with open("characters.txt", "r") as file:
        # Read the contents of the file into a single string
        contents = file.read()

    logging.basicConfig(filename="DataPrep.StdFonts.log", level=logging.DEBUG)

    for i, character in enumerate(contents):
        logging.debug(f"Processing {i} {character} \n")
        char_path = os.path.join(data_path, character)
        if not os.path.exists(char_path):
            os.mkdir(char_path)
        else:
            print(f"Directory exists {i} {character}")
        for j, font in enumerate(fonts):
            img = ChartoImg(character, font=ImageFont.truetype(font, size=200))
            name = "Char-" + str(i) + "-Font-" + str(j) + ".png"
            Image.fromarray(img).convert("L").save(os.path.join(char_path, name))
