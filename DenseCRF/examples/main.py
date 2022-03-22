# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import PIL.Image as Image
import pydensecrf.densecrf as dcrf


# codes of this function are borrowed from https://github.com/Andrew-Qibin/dss_crf
def crf_refine(img, annos):
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')




img_path = './image/'
mask_path = './mask/'
output_path = './output/'
List = os.listdir(img_path)

if (not os.path.exists(output_path)):
    os.makedirs(output_path)

for name in List:
    print(name)
    img_name = img_path+name
    img = Image.open(img_name).convert('RGB')

    mask_name = mask_path+name.replace('jpg','png')
    mask = cv2.imread(mask_name,0)


    prediction = crf_refine(np.array(img), mask)

    prediction = (prediction - prediction.min()) /(prediction.max() - prediction.min() + 1e-8)

    cv2.imwrite(output_path+name, prediction*255)



