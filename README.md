# Overview

This is the official implementation of our paper "GeoDTR+: Toward generic cross-view geolocalization via geometric disentanglement". This paper was driven by our previous research "Cross-view geo-localization via learning disentangled geometric layout correspondence"([paper](https://ojs.aaai.org/index.php/AAAI/article/view/25457), [code](https://gitlab.com/vail-uvm/geodtr)).

In this work, we extend our previous GeoDTR into GeoDTR+ with an enhanced GLE module that better models the correlations among visual features. To fully explore the LS techniques from our preliminary work, we further propose Contrastive Hard Samples Generation (CHSG) to facilitate model training. For detail information, please refer to our [paper](https://arxiv.org/abs/2308.09624).

# Usage

## Prerequisites
1. numpy
2. Pytorch >= 1.13
3. torchvision >= 0.12
4. tqdm
5. scipy
6. PIL

## datasets

### CVUSA

- We obtain the permission of CVUSA dataset from the owner by submit the [MVRL Dataset Request Form](https://mvrl.cse.wustl.edu/datasets/cvusa/).
- Please refer to the repo: [https://github.com/viibridges/crossnet](https://github.com/viibridges/crossnet)

### CVACT

- We obtain the permission of CVACT dataset by contacting the author directly.
- Please refer to the repo: [https://github.com/Liumouliu/OriCNN](https://github.com/Liumouliu/OriCNN)

### VIGOR

- We obtain the permission of VIGOR dataset from the owner by submit the [Questionnaire Form](https://github.com/Jeff-Zilence/VIGOR?tab=readme-ov-file).


## Training
### CVUSA and CVACT

```bash
python train.py \
--dataset CHOOSE_BETWEEN_CVUSA_OR_CVACT \
--save_suffix GIVE_A_SAVING_NAME \
--data_dir PATH_TO_YOUR_DATASET \
--geo_aug strong \
--sem_aug strong \
--backbone CHOOSE_BETWEEN_resnet_OR_convnext \
--bottleneck
```

Toggling `--cf` for counterfactual learning schema, `--mutual` for contrastive hard sample generation, `--no_polar` for disable polar transformation, `--verbose` for progressive bar.

This code will create a folder named by `--save_suffix` to store weights. To resume training you can set `--resume_from` by giving the folder name. All parameters will be set automatically.

### VIGOR

Training on VIGOR dataset basically is the same as on CVUSA and CVACT as described above. But please run `python train_vigor.py` instead. To be noticed, VIGOR does not support polar transformation.

## Testing

For testing on CVUSA and CVACT please use `python test.py --model_path MODEL_FOLDER --dataset CVUSA_OR_CVACT`.

For testing on VIGOR please use `python test_vigor.py --model_path`. 

Both script supports toggling `--verbose` for progressive bar.

## Pre-trained weights

The pre-trained weights can be found [here](https://drive.google.com/file/d/1JeX4378JDRlBIQGQ6ZVOYmQ_02YjZgKH/view?usp=sharing). 

After extracting the file, there should be six folders containing pre-trained weights for CVUSA w/ polar transformation (CVUSA), CVACT w/ polar transformation (CVACT), CVUSA w/o polar transformation (CVUSA_NP), CVACT w/o polar transformation (CVACT_NP), VIGOR same-area (VIGOR_same), and VIGOR cross-area (VIGOR_cross). Please following previous guidelines to evaluate the model.

# Citation

```
@ARTICLE{10636837,
  author={Zhang, Xiaohan and Li, Xingyu and Sultani, Waqas and Chen, Chen and Wshah, Safwan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={GeoDTR+: Toward Generic Cross-View Geolocalization via Geometric Disentanglement}, 
  year={2024},
  volume={},
  number={},
  pages={1-19},
  keywords={Feature extraction;Layout;Training;Correlation;Transformers;Data mining;Accuracy;Visual Geolocalization;Cross-view Geolocalization;Image Retrieval;Metric Learning},
  doi={10.1109/TPAMI.2024.3443652}}
  ```
