# Introduction

This code is the implement of our paper "Improving Alzheimer’s Disease Diagnosis with Multi-Modal PET Embedding Features
by a 3D Multi-task MLP-Mixer Neural Network", including both the proposed model and other competing methods.

## Data Preparation

1. The sMRI data should be arranged like "ADNI/subject_id/viscode.nii".
2. Using the CAT12 toolbox for SPM to preprocess the sMRI.
3. Specify the sMRI data path in utils/datasets.py
4. Download ADNIMERGE.csv, UCBERKELEYAV45_01_14_21.csv, and UCBERKELEYFDG_05_28_20.csv from the ADNI website and store
   them in data/.

## Example

### To train the regression module

```shell
python main.py --cuda_index 0  --method RegMixer --dataset ADNI_PET --clfsetting regression --batch_size 8 --n_epochs 100 \
--save_path regmodel.pth
```

### To train the classification module

```shell
python main.py --cuda_index 0 --method ClfMixer --clfdataset ADNI_dx --clfsetting CN-AD --batch_size 8 --n_epochs 100 \ 
--save_path clfmodel_cn_ad.pth
```

### To train the whole model

```shell
REGPREPATH=regmodel.pth CLFPREPATH=clfmodel_cn_ad.pth \
python main.py --method FuseMixer --dataset ADNI_dx --clfsetting CN-AD --batch_size 8 --n_epochs 100

CLFPREPATH=classfication_pretrain_model_path REGPREPATH=regression_pretrain_model_path \ 
python main.py --method FuseMixer --dataset ADNI_dx --clfsetting sMCI-pMCI --batch_size 8 --n_epochs 100
```

### Citation

If you use this code, please cite our paper:

```text
@ARTICLE{10137746,
  author={Zhang, Zi-Chao and Zhao, Xingzhong and Dong, Guiying and Zhao, Xing-Ming},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Improving Alzheimer's Disease Diagnosis with Multi-Modal PET Embedding Features by a 3D Multi-task MLP-Mixer Neural Network}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/JBHI.2023.3280823}}
```
