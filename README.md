# RestoreCAE-Zi

A neural network to restore inscriptions from erosion and damages based on standard font. 

The network architecture is customized based on Autoencoder and [ConvNeXt](https://arxiv.org/abs/2201.03545). The input is class-informed by providing an additional channel based on standard font along with inscription images.

The encoder is based on ConvNeXt-T and has around 30M parameters, and the decoder structure is the reversed counterpart. The latent space dimension is 512 by default. The network was trained on 4 A100 for 200 epochs.

This work is mostly for demo purpose. More comprehensive data preparation and ablation studies wait to be done to improve generalization and explore the complexity-performance trade-off. 

Environment: Python3.9 + requirements.txt 

## Project Background (if you are interested)
Ancient inscriptions, spanning hundreds or even thousands of years, often bear the marks of additional noise and defects resulting from natural erosion or deliberate damage. This is part of my efforts to restore inscriptions and ancient hand writings employing machine learning techniques. As an example, the left panel is the digitized image of a character (Zi) extracted from an inscription (普觉国师碑). The historical inscription dates back to approximately 1289 during the Southern Song Dynasty and is featured by the calligraphy style of Xizhi Wang(王羲之集字). The middle panel is the contemporary standard writing, and the right one is the restored version generated from the digitized inscription using this neural network.

![Example Xizhi's handwriting](./example-images/xizhi-example.png)

## 1. Results

Here are a few examples of restored handwriting images. The left two columns (processed and standard writings) are the input of the neural network; the third column is the ground truth; and the last column is the restoration result given by the neural network. 

<img src="./example-images/summarizedRestore.png" alt="drawing" width="500"/>

## 2. Training

The checkpoint file is available [here](https://drive.google.com/file/d/1m8e-eeI0zy6sOcmC2_Z1ooOk1Wlz6gwu/view?usp=sharing). The [dataset](https://drive.google.com/file/d/15_tXRqRtOpTFuoFpXNOtbrBWic0IqzRg/view?usp=sharing) contains 12 hand writings for each of the 3751 characters (Zi). 

For data augmentation, each input image goes through a random collection of image processing methods, including erosion and dilation, random rectangle painting, blurring, and random noise. 

The dataset can be further expanded for rich variety in the future given more efforts and resources. 

Training script: python3.9 -m torch.distributed.launch --nproc_per_node=N main_denoise_cae.py  

## 3. Testing