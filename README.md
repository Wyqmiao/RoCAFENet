# RoCAFENet
This project provides the code and results for 'Robust Salient Object Detection in Optical Remote Sensing Images via Multi-Scale Contextual Attention and Feature Enhancement'

# Network Architecture
   <div align=center>
   <img src="https://github.com/Wyqmiao/RoCAFENet/blob/main/images/RoCAFENet.jpg">
   </div>
   
   
# Requirements
   python 3.8 + pytorch 1.9.0

# Saliency maps
   We provide saliency maps of our RoCAFENet on three datasets in './RoCAFENet_saliencymapT.rar' 
   
# Training
   Download [pvt_v2_b2.pth](https://pan.baidu.com/s/1U6Bsyhu0ynXckU6EnJM35w) (code: sxiq), and put it in './model/'. 
   
   Modify paths of datasets, then run train.py.

Note: Our main model is under './model/RoCAFENet.py'

# Quantitative comparison
Quantitative comparison with state-of-the-art methods on the ORSSD and EORSSD datasets.
   <div align=center>
   <img src="https://github.com/Wyqmiao/RoCAFENet/blob/main/images/table1.jpg">
   </div>
   
Quantitative comparison with state-of-the-art methods on the ORSI-4199 dataset.
   <div align=center>
   <img src="https://github.com/Wyqmiao/RoCAFENet/blob/main/images/table2.jpg">
   </div>
   
# Visualization
Visualization comparing RoCAFENet with eight advanced methods on the ORSI-4199 dataset.
   <div align=center>
   <img src="https://github.com/Wyqmiao/RoCAFENet/blob/main/images/Visualization comparison.jpg">
   </div>
