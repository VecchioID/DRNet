# DRNet
Pytorch code for DRNet presented in paper, if you find our work helps for you, pleae cite our work as follows:

# Citation
```
@inproceedings{zhao2024learning,
  title={Learning Visual Abstract Reasoning through Dual-Stream Networks},
  author={Zhao, Kai and Xu, Chang and Si, Bailu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={15},
  pages={16979--16988},
  year={2024}
}
```

We will release our full codes with another publication! Please use checkpoint files to replicate our results in our paper!

Run the following command to reproduce the results present in our paper:
```
python train.py --dataset_path <path to data> --multi_gpu 2 -dataset pgm --batch_size 256 --epochs 401
```
# Requirements
Tested on Linux
- Python 3.9
- Pytorch 1.12.0
- numpy 1.19.2
- scipy 1.5.2
- matplotlib 3.3.2
- scikit-image 0.17.2
- tqdm 4.50.2
- torchvision 1.12.0
