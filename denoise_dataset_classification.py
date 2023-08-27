"""
Prepare painted data from DataPair into Tensors
"""
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from PIL import Image
import datetime
from torchvision import datasets, transforms
from torch import Tensor
from torch.utils.data import TensorDataset, random_split
import random

from cae import CAE


from PaintingMethods import img_processing_pool

"""
def painting(img):
    # use PaintingMethods to process imgs
    chosen_functions = random.sample(img_processing_pool, k=3)
    for processing in chosen_functions:
        img = processing(img)
    return img
"""

class DataPair:
    def __init__(self, img_input, img_output):
        self.input = img_input
        self.output = img_output

        
class PaintingTransform(object):
    def __init__(self, steps=3):
        self.step = steps

    def __call__(self, img):
        # Apply your custom transformation to the input image
        img_array = np.array(img)
        chosen_functions = random.sample(img_processing_pool, k=self.step)
        for processing in chosen_functions:
            img_array = processing(img_array)
        return Image.fromarray(img_array, mode='L')

"""
def prep_tensor(data_path):
    # resize into (224, 224) and add painting
    files = os.listdir(data_path)
    data_files = [file for file in files if "pickle" in file]

    resize = transforms.Resize((224, 224))
    transform = transforms.ToTensor()

    in_tensor_list = []  # : Tensor = torch.empty((1, 2, 224, 224))
    out_tensor_list = []  # : Tensor = torch.empty((1, 1, 224, 224))

    for data_file in data_files:
        with open(os.path.join(data_path, data_file), "rb") as f:
            img_dict = pickle.load(f)
            for key in img_dict:
                print(f"Working on {key}")
                for data_pair in img_dict[key]:
                    label = data_pair.input  # note that input is standard chars, output is calligraphy chars
                    output = data_pair.output
                    painted = painting(output)

                    painted_image = resize(Image.fromarray(painted))
                    label_image = resize(Image.fromarray(label))
                    input_image = Image.merge('LA', (painted_image, label_image))  # merge painted and label into input
                    out_image = resize(Image.fromarray(output))

                    in_tensor_item = transform(input_image).unsqueeze(0)
                    out_tensor_item = transform(out_image).unsqueeze(0)


                    in_tensor_list.append(in_tensor_item)  # = torch.cat((in_tensor, in_tensor_item), dim=0)
                    out_tensor_list.append(out_tensor_item)  # = torch.cat((out_tensor, out_tensor_item), dim=0)
    in_tensor = torch.cat(in_tensor_list, dim=0)
    out_tensor = torch.cat(out_tensor_list, dim=0)

    return in_tensor, out_tensor


def build_dataset(in_tensor, out_tensor, split_ratio):

    # split tensor into train and val
    dataset = TensorDataset(in_tensor, out_tensor)
    length = len(dataset)
    val = int(length * split_ratio)
    train_data, val_data = random_split(dataset, [length-val, val])
    return train_data, val_data

def read_tensor(data_file):

    # read tensor files for cae
    in_tensor = torch.load('in_' + data_file)
    out_tensor = torch.load('out_' + data_file)
    return in_tensor, out_tensor
"""


def build_dataset(args):
    transform = build_transform(step=3)
    """
    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")
    """
    if args.data_set == "image_folder":
        root = args.data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(step=3):
    painting = PaintingTransform(step)
    t = [transforms.Grayscale(), painting, transforms.ToTensor()]
    return transforms.Compose(t)


def get_args_parser():
    parser = argparse.ArgumentParser('Denoise dataset module', add_help=False)
    parser.add_argument('--data_set', default='image_folder', type=str)
    parser.add_argument('--data_path', default='DicData', type=str)
    parser.add_argument('--eval_data_path', default='DicData', type=str)
    parser.add_argument('--nb_classes', default=3751, type=int)
    # there is a mismatch in num of classes, 3751 vs 3753, resolved
    return parser


def main(args):
    print(args)
    dataset, args.nb_classes = build_dataset(args=args)
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True,
        batch_size=9)

    for batch_data in data_loader:
        # Extract inputs and labels from the batch
        inputs, labels = batch_data
        break
    grid_img = torchvision.utils.make_grid(inputs, nrow=3)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Noised dataset display', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

    """
    data_path = os.getcwd()
    sub_folder = 'StdData'
    data_path = os.path.join(data_path, sub_folder)


    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d")
    in_tensor, out_tensor = prep_tensor(data_path)
    torch.save(in_tensor, 'in_' + timestamp + '.pt')
    torch.save(out_tensor, 'out_' + timestamp + '.pt')
    """




