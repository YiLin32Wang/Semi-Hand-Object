import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

import torch.nn as nn

from src import model
from src import util
#from src.body import Body
from src.hand import Hand

import os
import torch
import json
import time

os.environ["CUDA_VISIBLE_DEVICE"] = '2'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class pose_detector(nn.Module):
    def __init__(self):
        super(pose_detector, self).__init__()
        self.hand_estimation = Hand('src/hand_pose_model.pth')
    
    def forward(self, images):
        peaks, values = self.hand_estimation(images)
        return peaks, values
    
def dump(pred_out_path, all_hand_peaks, all_hand_peaks_values, all_hand_names):
    xy_pred_list = [x.tolist() for x in all_hand_peaks]
    value_pred_list = [x.tolist() for x in all_hand_peaks_values]
    name_list = [x.tolist() for x in all_hand_names]

    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xy_pred_list,
                value_pred_list,
                name_list
            ], fo)
    fo.close()
    print(f"Dumped {len(all_hand_peaks)} predictions to {pred_out_path}")

    

if __name__ == "__main__":
    dataset_root = "/home/datassd/yilin/HO3D"
    mode = "train"
    list_root = "/home/datassd/yilin/Codes/Hand/Semi-Hand-Object/ho3d-process"
    list_path = os.path.join(list_root, "train.txt")
    save_root = list_root
    #list_path = "/home/datassd/yilin/HO3D/evaluation.txt"

    hand_estimation = pose_detector()
    all_hand_peaks = []
    all_hand_peaks_values = []
    all_hand_names = []
    start = time.time()

    with open(list_path) as f:
        img_list = [line.strip() for line in f]

    
    for idx, seq in enumerate(img_list):
        seqName, id = seq.split("/")
        img_path = os.path.join(dataset_root, mode, seqName, "rgb", id+'.jpg')
        if not os.path.exists(img_path):
            continue
        oriImg = cv2.imread(img_path)
        hand_peaks = []
        peaks, values = hand_estimation(images=oriImg)
        hand_peaks.append(peaks)
        if idx%50 == 0:
            save_img_path = os.path.join(list_root, "visual")
            os.makedirs(save_img_path, exist_ok=True)
            canvas = copy.deepcopy(oriImg)
            canvas = util.draw_handpose(canvas, hand_peaks)
            plt.imshow(canvas[:, :, [2, 1, 0]])
            plt.axis('off')
            plt.savefig(os.path.join(save_img_path,seqName+'_'+id+'.jpg'))
            print('{0} demo out saved!'.format(seqName))
            plt.close()
            #print('Sample {0} / {0} processed. Time: {0:.3f}\t'.format(idx, len(img_list), time.time() - start))
            print(f"Sample {idx} / {len(img_list)} processed. Time:{time.time() - start:.3f}")
            
        all_hand_peaks.append(peaks)
        all_hand_peaks_values.append(values)
        all_hand_names.append(np.array([seqName+'_'+id]))
        #break
    
    detect_out_path = save_root+'/train_2djoint.json'
    #os.makedirs(detect_out_path, exist_ok=True)
    dump(detect_out_path, all_hand_peaks, all_hand_peaks_values, all_hand_names)
    