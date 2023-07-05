# VRNet
Source code for RPM problems. 

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

# Citation
```
@inproceedings{zhang2019raven,
  title={Raven: A dataset for relational and analogical visual reasoning},
  author={Zhang, Chi and Gao, Feng and Jia, Baoxiong and Zhu, Yixin and Zhu, Song-Chun},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5317--5327},
  year={2019}
}

@inproceedings{benny2021scale,
  title={Scale-localized abstract reasoning},
  author={Benny, Yaniv and Pekar, Niv and Wolf, Lior},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12557--12565},
  year={2021}
}

@inproceedings{hu2021stratified,
  title={Stratified rule-aware network for abstract visual reasoning},
  author={Hu, Sheng and Ma, Yuqing and Liu, Xianglong and Wei, Yanlu and Bai, Shihao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={2},
  pages={1567--1574},
  year={2021}
}

@inproceedings{barrett2018measuring,
  title={Measuring abstract reasoning in neural networks},
  author={Barrett, David and Hill, Felix and Santoro, Adam and Morcos, Ari and Lillicrap, Timothy},
  booktitle={International conference on machine learning},
  pages={511--520},
  year={2018},
  organization={PMLR}
}

@article{nie2020bongard,
  title={Bongard-logo: A new benchmark for human-level concept learning and reasoning},
  author={Nie, Weili and Yu, Zhiding and Mao, Lei and Patel, Ankit B and Zhu, Yuke and Anandkumar, Anima},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={16468--16480},
  year={2020}
}
```