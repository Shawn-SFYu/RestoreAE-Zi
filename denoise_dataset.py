import argparse
import os
import os.path
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets.folder import IMG_EXTENSIONS, DatasetFolder
from gen_base_img import char2img
from denoise_dataset_classification import get_args_parser, build_transform

img2tensor = ToTensor()

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert('L')

class DenoiseFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::
           root/dog/xxx.png
           root/dog/xxy.png
           root/dog/[...]/xxz.png

           root/cat/123.png
           root/cat/nsdf3.png
           root/cat/[...]/asd932_.png

       This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
       the same methods can be overridden to customize the dataset.

       Args:
           root (string): Root directory path.
           transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
           target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
           loader (callable, optional): A function to load an image given its path.
           is_valid_file (callable, optional): A function that takes path of an Image file
               and check if the file is a valid file (used to check of corrupt files)

        Attributes:
           classes (list): List of the class names sorted alphabetically.
           class_to_idx (dict): Dict with items (class_name, class_index).
           imgs (list): List of (image path, class_index) tuples
       """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root=root,
            loader=loader,  # default pil_loader
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,  # default None
            is_valid_file=is_valid_file,
        )
        # self.imgs = self.samples # samples (list): List of (sample path, class_index) tuples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, original)
            sample noisy image + standard font
            original denoised image aka original image.
        """
        path, target = self.samples[index]  # samples (list): List of (sample path, class_index) tuples
        character = os.path.basename(os.path.dirname(path))
        original = self.loader(path)  # original image
        if self.transform is not None:
            noised = self.transform(original) # remove 255 *
            noised = np.array(noised, np.float32).squeeze()
        standard = np.array(char2img(character, cvt_traditional=True), dtype=np.float32) / 255  # remove 255
        sample = np.stack((noised, standard), axis=0)
        sample = torch.from_numpy(sample)
        original = img2tensor(original)  # torch.squeeze(img2tensor(original))  # torch.squeeze((255 * img2tensor(original)).to(torch.uint8))

        return sample, original

    def __len__(self) -> int:
        return len(self.samples)


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
        dataset = DenoiseFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


if __name__=='__main__':
    parser = argparse.ArgumentParser('Denoise Dataset', parents=[get_args_parser()])
    args = parser.parse_args()
    dataset, args.nb_classes = build_dataset(args=args)
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True,
        batch_size=1)

    for batch_data in data_loader:
        # Extract inputs and labels from the batch
        input_img, original = batch_data
        break
    #  input_array = np.array(input_img)
    noisy = input_img[:, 0, :, :]
    standard = input_img[:, 1, :, :]
    print(f"noisy shape {noisy.shape}")
    print(f"original shape {original.shape}")
    original = torch.squeeze(original, dim=0)
    print(f"original shape squeeze {original.shape}")

    grid_img = torchvision.utils.make_grid([noisy, standard, original], nrow=3, padding=4)
    plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    plt.show()