# Insect Classification - Image Feature extraction
Extract the features from insects' images to be used together with barcoding features for taxonomy applications.

## Requirements

 > Ubuntu 20.04

We suggest creating a virtual environment with the packages used in our code.

Please note that we used cuda-11.8. The version is not strictly necessary, but for convenience, please follow [these instructions](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local) to install.


```
 # create virtualenv
 python3.9 -m venv insectsvenv
 source insectsvenv/bin/activate

 # install PyTorch first
 pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

 # install all the other requirements
 pip install -r requirements.txt

```

## Download the data
Download [insect_dataset.mat](https://zenodo.org/records/14277812) from Zenodo.


## Checkpoints

We provide [the checkpoints](https://zenodo.org/records/15034519) at different stages of the training, in order to facilitate the usage and reproduce our results.

| Stage        | Checkpoint                          | Link                     |
|---------     |------------                         |--------------------------|
| DNA model    | CNN_DNA_weights_for_unseen.pt       | [Download](https://zenodo.org/records/15034519/files/CNN_DNA_weights_for_unseen.pt?download=1)                        |
| Generator    | generatorPretrainedReACGAN25.pt     | [Download](https://zenodo.org/records/15034519/files/generatorPretrainedReACGAN25.pt?download=1)   (optional)         |
| Generator    | generatorFinetunedReACGAN12.pt      | [Download](https://zenodo.org/records/15034519/files/generatorFinetunedReACGAN12.pt?download=1)   (optional)         |
| Discriminator| discriminatorPretrainedReACGAN25.pt | [Download](https://zenodo.org/records/15034519/files/discriminatorPretrainedReACGAN25.pt?download=1)   (optional)         |
| Discriminator| discriminatorFinetunedReACGAN12.pt  | [Download](https://zenodo.org/records/15034519/files/discriminatorFinetunedReACGAN12.pt?download=1)            |

## GAN model training and feature extraction

#### Pretrain the model (optional)
 - Download the data from [Zenodo](https://zenodo.org/records/14577906).

We pretrained the discriminator using the Jupyter Notebook PreTrainReACGAN.ipynb

#### Train the model (optional)
```
python GanScript.py -t -b 32 --dataset-path matlab_dataset/insect_dataset.mat --save-weights-path checkpoints --read-weights-path generatorPretrainedReACGAN25.pt
```

#### Extract features using a trained discriminator model
```
python GanScript.py -f -b 32 --dataset-path matlab_dataset/insect_dataset.mat --read-weights-path checkpoints/discriminatorFinetunedReACGAN12.pt
```
This will produce the file <b>all_image_features_gan.mat</b> that will be used in the Matlab code.

## DNA model training and feature extraction

#### Train the model (optional)
```
python DnaScript.py -t -b 32 --save-weights-path checkpoints --dataset-path=../matlab_dataset/insect_dataset.mat
```

#### Extract features using a trained discriminator model
```
python DnaScript.py -f -b 32 --read-weights-path CNN_DNA_weights_for_unseen.pt --dataset-path=../matlab_dataset/insect_dataset.mat
```
This will produce the file <b>all_dna_features.mat</b> that will be used in the Matlab code.


## If you use this work, please cite our paper:
```tex
@article{nanni2025algorithms,
    author = {Nanni, Loris and Gobbi, Matteo De and Junior, Roger De Almeida Matos and Fusaro, Daniel},
    title = {Advancing Taxonomy with Machine Learning: A Hybrid Ensemble for Species and Genus Classification},
    journal = {Algorithms},
    volume = {18},
    year = {2025},
    number = {2},
    article-number = {105},
    url = {https://www.mdpi.com/1999-4893/18/2/105},
    issn = {1999-4893},
    doi = {10.3390/a18020105}
}
```