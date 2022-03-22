## DSU Source Code

Source repository for our paper entilted "[Promoting Saliency From Depth: Deep Unsupervised RGB-D Saliency Detection](https://openreview.net/pdf?id=BZnnMbt0pW)" accepted by ICLR 2022 conference.

------



## Pre-Implementation

1. ```vim ./DenseCRF/README.md ```
2. **DenseCRF Installation**: Refer to [DenseCRF Readme.md](https://github.com/jiwei0921/DSU/blob/main/DenseCRF/README.md), and run demo successfully.
3. **Pytorch Environment**: Run ```conda install pytorch torchvision cudatoolkit=10.1 -c pytorch```.
4. Run ```pip install tqdm```.
5. Run ```pip install pandas```.
6. Run ```pip install tensorboardX```.
7. Run ```pip install fairseq```. Possible Question "SyntaxError: invalid syntax", please see FAQ-Q1 below.
8. Run ```pip install scipy```.
9. Run ```pip install matplotlib```.


### Dataset & Evaluation
1. The dataset used in this paper you can download directly ([Baidu Cloud (Passworde: bn1t)](https://pan.baidu.com/s/1WBi3-YlL8-d0kz-k2CudTA) or [Google Drive](https://drive.google.com/file/d/1oXyLs_Pki9qcGx4HnAtDh1FNbyhG4D-X/view?usp=sharing)), including training set with initial pseudo-labels, and test set. 
2. We use [this toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) for evaluating all SOD models.


------


### Our DSU Implementation

1. Modify the path of dataset in ```python DSU_test.py``` and ```python DSU_train.py```.
2. **Inference stage (Test your own dataset)**: ```python DSU_test.py```; Using Pre-trained Model in ```./ckpt``` ([Baidu Cloud (Passworde: vs85)](https://pan.baidu.com/s/1GbHR4V3jzqh1SGaQopIJGw) or [Google Drive](https://drive.google.com/file/d/1osp-8nEx_cAY9mjhaC9OJRTIQSP0irdr/view?usp=sharing)).  
3. **Training stage**: ```CUDA_VISIBLE_DEVICES=0 python DSU_train.py```                             
4. Check the log file: ```cat ./result.txt```


### Saliency Results

Our deep unsupervised saliency results can be approached in [Baidu Cloud (Passworde: m10a)](https://pan.baidu.com/s/1oPJjR2apBvnbUkmNokr3CQ) or [Google Drive](https://drive.google.com/file/d/1VwvTZFwRUtoEdymv5RzxywWuBmn4z7Xx/view?usp=sharing).
If you want to use our JSM to test on your own dataset, you can load our pretrained ckpt and run ```python demo_test.py``` directly.



### Bibtex
```
@inproceedings{
ji2022promoting,
title={Promoting Saliency From Depth: Deep Unsupervised {RGB}-D Saliency Detection},
author={Wei Ji and Jingjing Li and Qi Bi and chuan guo and Jie Liu and Li Cheng},
booktitle={International Conference on Learning Representations},
year={2022}
}
```

### Contact Us
If you have any questions, please contact us ( wji3@ualberta.ca ).


---
+ #### FAQ

**Question1**: When installing ```fairseq```， post an 'SyntaxError: invalid syntax' 

Answer1: You can directly update python version, e.g., ```conda install python=3.7```. More details can be found [in this channel](https://github.com/pytorch/fairseq/issues/55).

**Question2**: You should replace the inplace operation by an out-of-place one. 

Answer2: This is because `*=` is not compatible with Python 3.9. `q *= self.scaling` -> `q = q * self.scaling`

