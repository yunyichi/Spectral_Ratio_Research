# Spectral_Ratio_Research

## Introduction
Natural scenes are illuminated by direct light (e.g., sunlight) and ambient light (which fills shadows). The ratio of these lights is crucial for tasks like brightening shadows and enhancing colors through gamma correction. We propose a method to estimate the spectral ratio from a single image.

## Dataset
We used a custom dataset of 4,097 raw images, annotating lit and shadowed regions for the same materials. Spectral ratios were calculated and transformed into linear and log spaces.

## Models
We trained two convolutional neural networks (CNNs): a custom CNN and another using transfer learning from GoogleNet.

### Custom CNN
- Folder: `CNN`
- How to run: download files from `CNN` folder, run `DeepLearningModel.py` with parameters for training, and run `test_script.py` for testing.

### GoogLeNet
- Folder: `Pretrained_GoogleNet`
- How to run: download files from `Pretrained_GoogleNet` folder, run `DeepLearningModel.py` with parameters for training, and run `test_script.py` for testing.

## Applications
Applying these spectral ratios to gamma correction improved image contrast, brightened shadows, and enhanced color vibrancy. Our app restores details in shadowed areas and enhances color contrast, outperforming basic gamma correction, even in complex scenes with scattered shadows.

### Demos
- **Demo 1**: Easy task with clear shadow boundaries (e.g., block building).
- **Demo 2**: Hard task with scattered shadows (e.g., green plant).

To run the demos, download the `Demos` folder, open `Demo_1` or `Demo_2` folder, and run `SRGammaCorrection.py`.