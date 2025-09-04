<div align="center">
<h1>RoCAFENet：Robust Salient Object Detection in Optical Remote Sensing Images via Multi-Scale Contextual Attention and Feature Enhancement </h1>
</div>

## 📆 News
This project provides the results for RoCAFENet.

## ⭐ Abstract
Salient object detection in optical remote sensing images (ORSI-SOD) faces numerous challenges due to complex and diverse scene structures, varying contexts, as well as shadow interference and low contrast caused by changes in imaging conditions. Existing models primarily utilize convolutional neural networks as encoders. However, their limited receptive fields hinder the capture of global information. Additionally, most models focus excessively on the contextual information of adjacent levels while neglecting the in-depth decoding and interaction of features at the same level. In particular, they lack a comprehensive exploration and integration of multiple features and contexts. Moreover, during the feature fusion stage, models typically rely solely on simple upsampling methods for prediction. To address these issues, we propose a Robust Multi-Scale Contextual Attention and Feature Enhancement Network (RoCAFE-Net), which mitigates information loss before cross-layer interaction by deeply decoding the features from each layer. The Spatial Displacement Self-Attention Module (SDSM) utilizes spatial interactions to effectively learn the diversity of salient object features and better model the absolute positional information of each block, as well as the relative positional information among blocks. The Channel and Spatial Detail Perception Module (CSDPM) emphasizes local features within the channel while preserving the fine-grained details of the salient object without altering inter-channel information. Finally, the Adaptive Feature Fusion Head (AFFH) fully integrates the intrinsic relationships between channels at different levels and gradually restores them to the same size as the input optical remote sensing image. Experiments on three ORSI-SOD benchmark datasets show that RoCAFE-Net significantly outperforms current state-of-the-art methods in performance.

## 📻 Network Architecture
   <div align=center>
   <img src="https://github.com/Wyqmiao/RoCAFENet/blob/main/images/RoCAFENet.jpg">
   </div>
The overall framework of RoCAFE-Net based on the encoder-decoder architecture mainly consists of four parts: the encoder uses pvt-v2-b2, followed by Spatial Displacement Self-Attention Module (SDSM), Channel and Spatial Detail Perception Module (CSDPM) and Adaptive Feature Fusion Head (AFFH).
   
## 🎮 Requirements
   python 3.8 + pytorch 1.9.0

## 🚀 Saliency maps
   We provide saliency maps of our RoCAFENet on three datasets in './RoCAFENet_saliencymapT.rar' 
   
## 🎈 Training
   Download [pvt_v2_b2.pth](https://pan.baidu.com/s/1F_W3XllVk8UGZ5oJpFWwHQ) (code: db7r), and put it in './model/'. 
   
   Modify paths of datasets, then run train.py.

Note: Our main model is under './model/RoCAFENet.py'

## 🖼️ Quantitative comparison
   <div align=center>
   <img src="https://github.com/Wyqmiao/RoCAFENet/blob/main/images/table1.jpg">
   </div>
   
   <div align=center>
   <img src="https://github.com/Wyqmiao/RoCAFENet/blob/main/images/table2.jpg">
   </div>
   
## 🎫 Visualization
   <div align=center>
   <img src="https://github.com/Wyqmiao/RoCAFENet/blob/main/images/Visualization comparison.jpg">
   </div>
Visualization comparison of prediction results from 8 methods, including 6 ORSI-SOD methods and 2 NSI-SOD methods. (a) ORSIs. (b) GT. (c) Our Method. (d) MJRBM. (e) ACCoNet. (f) MCCNet. (g) SeaNet. (h) UG2L. (i) SDNet. (j) VST. (k) ICON.
