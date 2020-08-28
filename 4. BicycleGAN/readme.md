## BicycleGAN : [Toward Multimodal Image-to-Image Translation](https://arxiv.org/abs/1711.11586)

### 0. Inference Result (After 1 Epoch)
<img src = './results/inference/BicycleGAN_Edges2Handbags_Results_001.png'>

### 1. Run the Codes
#### 1) Download Datasets
```
sh download_dataset.sh edges2handbags
```
#### 2) Directory
Check the directory corresponds to the following.
```
+---[data]
|   \---[edges2handbags]
|       \----[test]
|               +---[]
|               |...
|               +---[]
|       \---[train]
|               +---[]
|               ...
|               +---[]
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
<img src = './results/samples/BicycleGAN_Losses_Epoch_1.png'>

### 3. Loss Plots
<img src = './results/plots/BicycleGAN_Losses_Epoch_1.png'>
