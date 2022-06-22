# Knowledge Distillation via the Target-aware Transformer (CVPR2022)

Codebase of our **TaT** on ImageNet.

## Overview
Executable code can be found in [examples/image_classification.py](examples/image_classification.py). The implementation of **TaT** is [AttnEmbed](torchdistill/models/special.py). The loss function [MaskedFM](torchdistill/losses/single.py) is decoupled with the model. 

### Note
1. This codebase currently do not support resume. However, it allows you to load a pre-trained model for specific purposes, i.e., distilling a contrastive learning model.
2. The classification model is wrapped with the learnable KD parameters. Please be careful on the model parameters you want to save.
## Customization
If you would like to customize your own model, please put all the learnable parameters on [here](torchdistill/models/special.py). And you can set up the calculation of the loss funcion on [here](torchdistill/losses/single.py). 

We use the Forward Hook to extract the intermediate representations. Just modify the yaml file to access the model layers of your interest. [This example notebook](demo/extract_intermediate_representations.ipynb) will give you a better idea of the usage. You may refer to our [config](configs/sample/ilsvrc2012/single_stage/tat/resnet18_from_resnet34_attn.yaml). 

## Examples

### Requirments
- Python 3.7
- pytorch 1.5
- einops
- ml-collection

### Before getting started
Please modify the ImageNet path of the [config](configs/sample/ilsvrc2012/single_stage/tat/resnet18_from_resnet34_attn.yaml).

We use 8 GPUs with 256 images per GPU.

### Training 
```
sh ./train_local.sh
```

### Testing
```bash
sh ./test_local.sh
```


## Issues / Contact
Feel free to create an issue if you get a question or just
email me ( sihao.lin@student.rmit.edu.au ). 

## Acknowledgement
This repo is built upon [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill). Thanks to [Yoshitomo](https://github.com/yoshitomo-matsubara).