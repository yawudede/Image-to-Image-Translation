## Unsupervised Attention-Guided GAN : [Unsupervised Attention-guided Image to Image Translation](https://arxiv.org/abs/1806.02311)

### 0. Inference Result (After 200 Epochs)
#### 1) Horse | Attention Maps | Generated Zebra
<img src = './results/inference/Horse2Zebra/UAG-GAN_Horse2Zebra_Results_031.png'>
<img src = './results/inference/Horse2Zebra/UAG-GAN_Horse2Zebra_Results_067.png'>


#### 2) Zebra | Attention Maps | Generated Horse
<img src = './results/inference/Zebra2Horse/UAG-GAN_Zebra2Horse_Results_019.png'>
<img src = './results/inference/Zebra2Horse/UAG-GAN_Zebra2Horse_Results_024.png'>


### 1. Run the Codes
#### 1) Download Datasets
```
sh download_dataset.sh horse2zebra
```
#### 2) Directory
Check if the directory corresponds to the following:
```
+---[data]
|   \---[horse2zebra]
|       \----[testA]
|               +---[n02381460_20.jpg]
|               |...
|               +---[n02381460_9260.jpg]
|       \----[testB]
|               +---[n02391049_80.jpg]
|               |...
|               +---[n02391049_10980.jpg]
|       \---[trainA]
|               +---[n02381460_2.jpg]
|               |...
|               +---[n02381460_9263.jpg]
|       \---[trainB]
|               +---[n02391049_2.jpg]
|               |...
|               +---[n02391049_11195.jpg]
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
Horse | Attention Maps | Generated Zebra | Zebra | Attention Maps | Generated Horse
<img src = './results/samples/UAG-GAN_Horse2Zebra_Epoch_050.png'>

### 3. Loss During Train Process
<img src = './results/plots/UAG-GAN_Losses_Epoch_200.png'>
