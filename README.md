# DEtection TRansformer Network (DETR) - Tensorflow 2

## Overview
Implementation of the **DETR** (**DE**tection **TR**ansformer) network (Carion, Nicolas, et al., 2020) in *Tensorflow 2*.
**This is still a work in progress!**

## References
- **Research Paper:** [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- **Original PyTorch Implementation:** [GitHub](https://github.com/facebookresearch/detr)

## Getting started
It is recommended to create a virtual environmet using [Anaconda](https://anaconda.org/) and [Pip](https://pypi.org/).

1. Create and start conda environmet:
```
conda create -n detr python=3.6 anaconda pip -y
conda activate detr
```

2. Install necessary libraries:
```
conda install tensorflow-gpu==2.x -y
pip install tensorflow-gpu==2.3.0rc1 tensorflow-addons==0.11.1 numpy scipy matplotlib imgaug opencv-python
```

3. Use the ```train_detr-tf2.ipynb``` notebook to start training and make inferences.

## Code organization
*In progress...*

## Known issues
- The model is currently not converging to good results. It may be due to the implemented loss function. So far, many investigations are beeing done around it.