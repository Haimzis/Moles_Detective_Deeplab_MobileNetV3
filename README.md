## Moles Detective MobileNetV3
Our project uses DeepLabV3 solutions for these Tasks:
- Semantic Segmentation - MobileNetV3.
- Classification - InceptionV3.

### ***Usage:*** 
1. Download the source code from github
2. Create your own Dataset directory and move it to `datasets`.
3. Create builder for your data, look at `build_moles_detector_data.py` for example.
4. configure `train.py` params
5. run `train.py`

### ***Our Problem Definition:*** 
**Goal**: recognize and classify Skin lesions.  
**Datasets**:

| Dataset Name| `Semantic Segmentation` |  `Classification` | `Num Of Samples` |
| :---: | :---: | :---: | :---: |
| ISIC2018 | V  | X | ~2000 |
| ISIC2019 | X | V | ~22000 |
| Our Own Dataset | V | X | ~3000 |

*we had to balance the classification dataset with augmentations.  

**HyperParams & Configuration**:

| Parameter Name| `Value` |  
| :---: | :---: |
| Optimizer |  Momentum |
| Learning Rate | 0.09 |
| LR Decay | 0.04 | 
| LR Decay Step | 750 |
| Steps | 110k |
| Batch Size | 16 |
| Image Scale | 250,250 | 
| Dropping | 87.5% |
| Quantize delay | 0

We practiced Quantization Aware Training.  
To get a quantizable model, for mobile inference.

**GPU**: GeForce GTX 960M  
**Results**:

| Task | `mean IOU` |  
| :---: | :---: |
| Semantic segmentation |  91% |
| Classification | 75% |