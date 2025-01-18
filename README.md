# RoCAFENet
This project provides the code and results for 'Robust Salient Object Detection in Optical Remote Sensing Images via Multi-Scale Contextual Attention and Feature Enhancement'

# Network Architecture
   <div align=center>
   <img src=" ">
   </div>
   
   
# Requirements
   python 3.8 + pytorch 1.9.0

# Saliency maps
   We provide saliency maps of our RoCAFENet on three datasets in './RoCAFENet_saliencymapT.rar' 
   
# Training
   Download [pvt_v2_b2.pth](https://pan.baidu.com/s/1U6Bsyhu0ynXckU6EnJM35w) (code: sxiq), and put it in './model/'. 
   
   Modify paths of datasets, then run train.py.

Note: Our main model is under './model/RoCAFENet.py'


# Pre-trained model and testing
1. Download the pre-trained models on [ORSSD](https://pan.baidu.com/s/1E6Llbauan4QXfgOvnrcP1w) (code: qga2), [EORSSD](https://pan.baidu.com/s/1dY_9UtDb5GVb9rFyBNDSCA) (code: ahm7), and [ORSI-4199](https://pan.baidu.com/s/1NPdsGBW72vGXgsZxYrJCcA) (code: 5h3u), and put them in './models/'.

2. Modify paths of pre-trained models and datasets.

3. Run test.py.

# Quantitative comparison
Quantitative comparison with state-of-the-art methods on the ORSSD, EORSSD and ORSI-4199 datasets.

# Visualization
Visualization comparing RoCAFENet with eight advanced methods on the ORSI-4199 dataset.
   
