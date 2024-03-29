# NP-Match

**NP-Match: When Neural Processes meet Semi-Supervised Learning**

Jianfeng Wang<sup>1</sup>, Thomas Lukasiewicz<sup>1</sup>, Daniela Massiceti<sup>2</sup>, Xiaolin Hu<sup>3</sup>, Vladimir Pavlovic<sup>4</sup> and Alexandros Neophytou<sup>5</sup>

*University of Oxford*<sup>1</sup>, *Microsoft Research*<sup>2</sup>, *Tsinghua University*<sup>3</sup>,  *Rutgers University*<sup>4</sup>, *Microsoft*<sup>5</sup>

In [ICML 2022](https://proceedings.mlr.press/v162/wang22s.html)

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

For the pertrained models, they were not saved. The pretrained models will be relased later once we have enough resources for training.

Update
-----
We also adapt Neural Processes to semi-supervised semantic segmentation task. Please refer to [NP-SemiSeg](https://github.com/Jianf-Wang/NP-SemiSeg) if you are interested in it.

Citation
-----------------

  ```
@inproceedings{wang2022np,
  title={NP-Match: When Neural Processes meet Semi-Supervised Learning},
  author={Wang, Jianfeng and Lukasiewicz, Thomas and Massiceti, Daniela and Hu, Xiaolin and Pavlovic, Vladimir and Neophytou, Alexandros},
  booktitle={International Conference on Machine Learning},
  pages={22919--22934},
  year={2022},
  organization={PMLR}
}
  ```
