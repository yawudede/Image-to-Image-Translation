## DiscoGAN : [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192)

### 0. Inference Result (After 100 Epochs)
<img src = './results/inference/DiscoGAN_Edges2Shoes_Results_001.png'>

### 1. Run the Codes
#### 1) Download Datasets
```
sh download_dataset.sh edges2shoes
```
#### 2) Directory
Check the directory corresponds to the following.
```
+---[data]
|   \---[edges2shoes]
|       \----[train]
|               +---[]
|               |...
|               +---[]
|       \---[val]
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
<img src = './results/samples/DiscoGAN_Edges2Shoes_Epoch_100.png'>

### 3. Loss Plots
<img src = './results/plots/DiscoGAN_Losses_Epoch_100.png'>
