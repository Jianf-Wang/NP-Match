# NP-Match

**NP-Match: When Neural Processes meet Semi-Supervised Learning**

Jianfeng Wang<sup>1</sup>, Thomas Lukasiewicz<sup>1</sup>, Daniela Massiceti<sup>2</sup>, Xiaolin Hu<sup>3</sup>, Vladimir Pavlovic<sup>4</sup> and Alexandros Neophytou<sup>5</sup>

*University of Oxford*<sup>1</sup>, *Microsoft Research*<sup>2</sup>, *Tsinghua University*<sup>3</sup>,  *Rutgers University*<sup>4</sup>, *Microsoft*<sup>5</sup>

In ICML 2022

[Arxiv](https://arxiv.org/abs/2207.01066)


Build
-----

please run with the following command:

```
conda env create -f NP-Match.yml
conda activate NP-Match
```

Experiment
-----
To run experiments on small datasets, e.g., CIFAR, STL-10, please run with their corresponding config files, stored in "config/npmatch/".

For example:

```
python npmatch.py --c config/npmatch/npmatch_cifar10_250_0.yaml
```

To run experiment on ImageNet, please prepare the dataset, and change the config file accordingly. Then, run with

```
python npmatch.py --c config/npmatch/npmatch_imagenet_100000.yaml
```

For the pertrained models, I did not save them, and I do not have enough computational resources at present. I will release the pretrained models once I have enough resources for training.

Citation
-----------------

Please cite our paper if this repo is helpful.
