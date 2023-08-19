# DenoiseCAE-Zi


 

The network architecture is customized and is largely based on [Autoencoder]() and [ConvNeXt](https://arxiv.org/abs/2201.03545). In this work, the encoder is based on ConvNeXt-T and has around 30M parameters, and the decoder structure is the reversed counterpart. The latent space dimension is 768. The network was trained on 4 A100 for 100 epochs.

This work is mostly for practice and demo purpose. Ablation studies wait to be done to explore the complexity-performance trade-off. 

Environment: Python3.9 + requirements.txt 

## 1. Results

## 2. Training

The checkpoint file is available [here](https://drive.google.com/file/d/1m8e-eeI0zy6sOcmC2_Z1ooOk1Wlz6gwu/view?usp=sharing). The [dataset](https://drive.google.com/file/d/15_tXRqRtOpTFuoFpXNOtbrBWic0IqzRg/view?usp=sharing) contains 12 hand writings for each of the 3751 characters (Zi). The dataset can be further expanded for rich variety in the future given more efforts and resources. 

Training script: python3.9 -m torch.distributed.launch --nproc_per_node=N main_denoise_cae.py  

## 3. Testing