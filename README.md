# Spectral_Ratio_Research

## Introduction
Natural scenes are illuminated by direct light (e.g., sunlight) and ambient light (which fills shadows). The ratio of these lights is crucial for tasks like brightening shadows and enhancing colors through gamma correction. We propose a method to estimate the spectral ratio from a single image.

## Dataset
We used a custom dataset of 4,097 raw images, annotating lit and shadowed regions for the same materials. Spectral ratios were calculated and transformed into linear and log spaces.

Link to the dataset: [Add link here]

## Models
We trained two convolutional neural networks (CNNs): a custom CNN and another using transfer learning from GoogleNet. The best model achieved a test angular error of 3.32°, demonstrating its effectiveness in spectral estimation.

### Custom CNN
- Folder: [Specify folder]
- File: [Specify file]
- How to run: [Provide instructions]

### GoogLeNet
- Folder: [Specify folder]
- File: [Specify file]
- How to run: [Provide instructions]

## Applications
Applying these spectral ratios to gamma correction improved image contrast, brightened shadows, and enhanced color vibrancy. Our app restores details in shadowed areas and enhances color contrast, outperforming basic gamma correction, even in complex scenes with scattered shadows.

### Demos
- **Demo 1**: Easy task with clear shadow boundaries (e.g., block building).
- **Demo 2**: Hard task with scattered shadows (e.g., green plant).

To run the demos, download the "Demo" folder, open Demo_1 or Demo_2, and run `SRGammaCorrection.py`.