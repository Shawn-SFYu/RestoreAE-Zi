# RestoreAE-Zi

Neural networks for restoring inscriptions and writings from damages with/wo standard font as reference. You can find more fun background information at the end of this document.

The network architectures are customized based on Autoencoder and [EfficientNetV2](https://arxiv.org/abs/2104.00298), [MobileNetV3](https://arxiv.org/abs/1905.02244), [ConvNeXt](https://arxiv.org/abs/2201.03545). These three autoencoder architectures are designed to be sysmetric, and the encoder and decoder should have similar capability. The autoencoder based on EfficientNetV2/MobileNetV3/ConvNeXt-T has around /4.5M/51M parameters respectively. The latent space dimension is 512 by default.

At the same time, asysmetric autoencoder is also popular: ablation studies have shown asysmetric autoencoders can provide state-of-art accuracy and at the same time gain extra benefits in throughput. I am also actively exploring asymetric models.

In the original design, the input is class-informed by providing an additional 'standard font' channel  along with inscription and writing images. The standard font is supposed to provide additional information about the inscrption or writing, and the corresponding character (the image in standard font) is also usually known before inscription restoration. Providing inscrption images as the only input also has protential advantages. considering that the 'standard' font actually change over time and can also vary. Enabling pretraining and then fine-tuning is another benefit for using inscription-only input.

This work is mostly for demo purpose. More comprehensive data preparation and ablation studies wait to be done to improve generalization and explore the complexity-performance trade-off. 

Environment: Python3.9 + requirements.txt 

## 1. Results

Here are a few examples of restored handwriting images. The left two columns (processed and standard writings) are the input of the neural network; the third column is the ground truth; and the last column is the restoration result given by the neural network. 

<img src="./example-images/summarizedRestore.png" alt="drawing" width="500"/>

## 2. Training

The checkpoint file is available [here](https://drive.google.com/file/d/1m8e-eeI0zy6sOcmC2_Z1ooOk1Wlz6gwu/view?usp=sharing). The [dataset](https://drive.google.com/file/d/15_tXRqRtOpTFuoFpXNOtbrBWic0IqzRg/view?usp=sharing) contains 12 hand writings for each of the 3751 characters (Zi). 

For data augmentation, each input image goes through a random collection of image processing methods, including erosion and dilation, random rectangle painting, blurring, and random noise. 

The dataset can be further expanded for rich variety in the future given more efforts and resources. 

Training script: python3.9 -m torch.distributed.launch --nproc_per_node=N main_denoise_cae.py  

## 3. Testing

Check out visualization/restore_eval.ipynb for restoration results.


## Fun Project Background
Ancient inscriptions, spanning hundreds or even thousands of years, often bear the marks of additional noise and defects resulting from natural erosion or deliberate damage. This is part of my efforts to restore inscriptions and ancient hand writings employing machine learning techniques. As an example, the left panel is the digitized image of a character (Zi) extracted from an inscription (普觉国师碑). The historical inscription dates back to approximately 1289 during the Southern Song Dynasty and is featured by the calligraphy style of Xizhi Wang (王羲之集字). The middle panel is the contemporary standard writing, and the right one is the restored version generated from the digitized inscription using this neural network.

![Example Xizhi's handwriting](./example-images/xizhi-example.png)