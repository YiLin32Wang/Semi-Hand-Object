import os
from torchvision.transforms import functional
from torch.utils import data
import random
import numpy as np
import cv2
from PIL import Image, ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch

from dataset import ho3d_util
from dataset import dataset_util
from utils import vis_utils

root = "/home/datassd/yilin/HO3D/"
train_label_root = "./ho3d-process-my/"
output_root = "./Outputs/"
os.makedirs(output_root, exist_ok=True)

jointsMapManoToSimple = [0, 
                        13, 14, 15, 16,
                        1, 2, 3, 17,
                        4, 5, 6, 18,
                        10, 11, 12, 19,
                        7, 8, 9, 20]

set_list = ho3d_util.load_names(os.path.join(train_label_root, "train.txt"))
print(f"Train images in v3:{len(set_list)}")
# camera matrix
K_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_K.json'))
#K_list = K_list[0]
# hand joints
joints_all_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_joint.json'))
joints_3d_list = joints_all_list[0]
joints_2d_list = joints_all_list[1]
# mano params
mano_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_mano.json'))
mano_pose_list = mano_list[0]
mano_trans_list = mano_list[1]
mano_beta_list = mano_list[2]
# obj landmarks
obj_all_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_obj.json'))
obj_p2d_list = obj_all_list[0]
obj_p3d_list = obj_all_list[1]

mode = "train"

for i, seq in enumerate(set_list):
    seqName, id = seq.split("/")
    img = ho3d_util.read_RGB_img(root, seqName, id, mode)
    mano_param = mano_pose_list[i]
    mano_param.extend(mano_beta_list[i])
    mano_param.extend(mano_trans_list[i])

    K = np.array(K_list[i], dtype=np.float32)
    joints_uv = np.array(joints_2d_list[i], dtype=np.float32)

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    imgAnno = vis_utils.showHandJoints(img_cv, joints_uv[jointsMapManoToSimple])
    filename = output_root + seqName + '_' + id + '.png'
    cv2.imwrite(filename, imgAnno)
    print(f"Visualized {filename}.")





