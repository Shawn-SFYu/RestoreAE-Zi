"""
Generate imgs from character based on ttf
"""
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from opencc import OpenCC
import numpy as np
cc = OpenCC('s2tw')


def char2img(character: str, font=None, cvt_traditional=False):
    assert len(character) == 1
    if not font:
        font = ImageFont.truetype('std_kai.ttf', size=200)
    if cvt_traditional:
        character = cc.convert(character)
    image = Image.new('RGB', (224, 224), color='black')
    draw = ImageDraw.Draw(image)

    # text_size = draw.textsize(character, font=font)
    text_location = (image.width // 2, image.height // 2)
    draw.text(text_location, character, font=font, fill='white', anchor='mm')
    # gray_array = np.asarray(image.convert('L'))

    return image.convert('L')


if __name__ == '__main__':
    image = Image.fromarray(char2img('丁'))

    # Convert the image to grayscale
    gray_image = image.convert('L')
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.show()


"""
def ChartoImg(character: str, font, simplified=True):
    assert len(character) == 1
    image = Image.new('RGB', (224, 224), color='black')
    draw = ImageDraw.Draw(image)
    if simplified:
        pass
    text_size = draw.textsize(character, font=font)
    text_location = ((image.width - text_size[0]) // 2, (image.height - text_size[1]) // 2)
    draw.text(text_location, character, font=font, fill='white')
    gray_array = np.asarray(image.convert('L'))

    return gray_array
"""
