# DenseCRF Installation
This is a ready-to-use **python DenseCRF** file for refining saliency maps using dense CRF, 
which is used in the NeurIPS 2021 paper "**Joint Semantic Mining for Weakly Supervised RGB-D Salient Object Detection**".
If you want to know more about CRF, you can refer to [here](http://graphics.stanford.edu/projects/drf/).    

   
## Dependencies
**Test successful on 2080Ti / RTX 3090 / Tesla P40 GPU**
1. Initial conda environment: ```conda create -n JSM python=3.7```.
2. ```conda activate JSM```
3. Install ***pydensecrf***: ```pip install git+https://github.com/lucasb-eyer/pydensecrf.git```
4. ```pip install --upgrade pip```
5. Install ***opencv***: ```pip install opencv-python```
6. Install ***numpy***:  ```pip install numpy```
7. Install ***PIL***: ```pip install Pillow```
8. Install ***skimage***: ```pip install -U scikit-image```


## Run test demo
**Check whether the installation is successful**
1. ```cd DenseCRF/examples/``` 
2. ```python main.py```
3. Check the outputs in `./output` file.


### Maybe Bug
1. pydensecrf/densecrf/include/Eigen/Core:22:10: **fatal error: ‘complex’ file not found**       
[#include](https://blog.csdn.net/u011599639/article/details/83934856)    
^~~~~~~~~    
1 warning and 1 error generated.        

**Command**: ``` conda install -c conda-forge pydensecrf ```


## Related Reference
1. https://github.com/Andrew-Qibin/dss_crf
2. https://github.com/jiwei0921/DenseCRF_refine_saliency-map


