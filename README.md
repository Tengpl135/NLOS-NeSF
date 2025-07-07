# Efficient Implicit Reconstruction of Hidden Object in Two-bounce Non-Line-of-Sight Imaging
This is the official pytorch implementation of "**Efficient Implicit Reconstruction of Hidden Object in Two-bounce Non-Line-of-Sight Imaging**.

<!--<h3 align="center"> -->

### [Project Page](https://github.com/Tengpl135/NLOS-NeSF.git)

## Requirements
Our experimental environment is
```
CUDA 11.7
pytorch 2.0.1
```
The requirements can be installed by
```
conda install --name nlos-neus --file environment.yml
conda activate nlos-nesf
```

## Data
The dataset is stored in the data folder

## Training
If you want to train a static scene:
```
python run_nlos.py --config configs/test.txt
```
Our dynamic scene reconstruction relies on the acquisition device to continuously update the dataset, and we give a version that replaces the dataset in training for testing.
```
python run_nlos_renewdata.py --config configs/test.txt
```
