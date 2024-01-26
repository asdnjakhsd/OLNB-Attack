from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from utils import permutation_utils

img = cv2.imread(
    "./testpic.jpg")

tensor_mat = permutation_utils.cvmat_to_tensor(img)
print("tensor mat size: ", tensor_mat.shape)
cv_mat = permutation_utils.tensor_to_cvmat(tensor_mat)
print("opencv mat size: ", cv_mat.shape)
square_size = 299

target_x, target_y = permutation_utils.get_square_pos(
     tensor_mat, 719, 559, square_size)

tensor_square = permutation_utils.get_square_tensor(
     tensor_mat, target_x, target_y, square_size)

res_square = permutation_utils.tensor_to_cvmat(tensor_square)

# print("original image-----------------------")
# print(img)

# print("conversed image-----------------------")
# print(cv_mat)


print(target_x,"   ", target_y)
cv2.imshow('img', permutation_utils.get_square_cvmat(img, target_x, target_y, square_size))
cv2.waitKey()
