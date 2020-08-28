## StarGAN : [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)

### 0. Inference Result (After 10 Epoch)
<img src = './results/inference/StarGAN_Aligned_CelebA_Results_0001.png'>

### 1. Run the Codes
#### 1) Download Datasets
```
sh download_dataset.sh
```
#### 2) Directory
Check the directory corresponds to the following.
```
+---[data]
|   \---[celeba]
|       \----[images]
|               +---[]
|               |...
|               +---[]
|       +---[list_attr_celeba.txt]
+---config.py
+---download_dataset.sh
|   ...
+---utils.py
```
#### 3) Train
```
python train.py
```
#### 4) Inference
```
python inference.py
```

### 2. Sample Generated During Training
<img src = './results/samples/StarGAN_Aligned_CelebA_Epoch_010.png'>

### 3. Loss Plots
<img src = './results/plots/StarGAN_Losses_Epoch_10.png'>
