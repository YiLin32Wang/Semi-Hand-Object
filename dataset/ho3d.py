import os
from torchvision.transforms import functional
from torch.utils import data
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch

from dataset import ho3d_util
from dataset import dataset_util

class HO3D(data.Dataset):
    def __init__(self, dataset_root, obj_model_root, train_label_root="./ho3d-process",
                 mode="evaluation", inp_res=512,
                 max_rot=np.pi, scale_jittering=0.2, center_jittering=0.1,
                 hue=0.15, saturation=0.5, contrast=0.5, brightness=0.5, blur_radius=0.5):
        # Dataset attributes
        self.root = dataset_root
        self.mode = mode
        self.inp_res = inp_res
        self.joint_root_id = 0
        self.jointsMapManoToSimple = [0, 13, 14, 15, 16,
                                      1, 2, 3, 17,
                                      4, 5, 6, 18,
                                      10, 11, 12, 19,
                                      7, 8, 9, 20]
        self.jointsMapSimpleToMano = np.argsort(self.jointsMapManoToSimple)
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        # object informations
        self.obj_mesh = ho3d_util.load_objects_HO3D(obj_model_root)
        self.obj_bbox3d = dataset_util.get_bbox21_3d_from_dict(self.obj_mesh)
        self.obj_diameters = dataset_util.get_diameter(self.obj_mesh)

        if self.mode == "train":
            self.hue = hue
            self.contrast = contrast
            self.brightness = brightness
            self.saturation = saturation
            self.blur_radius = blur_radius
            self.scale_jittering = scale_jittering
            self.center_jittering = center_jittering
            self.max_rot = max_rot

            self.train_seg_root = os.path.join(train_label_root, "train_segLabel")

            self.mano_params = []
            self.joints = []
            self.joints_uv = []
            self.obj_p2ds = []
            self.K = []
            # training list
            self.set_list = ho3d_util.load_names(os.path.join(train_label_root, "train.txt"))
            print(f"Original train images in v2:{len(self.set_list)}")
            set_id_list = []
            for idx, seq in enumerate(self.set_list):
                seqName, id = seq.split("/")
                img_path = os.path.join(self.root, self.mode, seqName,'rgb', id + '.jpg')
                if not os.path.exists(img_path):
                    self.set_list.pop(idx)
                    continue
                set_id_list.append(idx)
            print(f"Revised train images in v3:{len(self.set_list)}")


            # camera matrix
            K_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_K.json'))
            # hand joints
            joints_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_joint.json'))
            # mano params
            mano_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_mano.json'))
            # obj landmarks
            obj_p2d_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_obj.json'))
            for i, idx in enumerate(set_id_list):
                K = np.array(K_list[idx], dtype=np.float32)
                self.K.append(K)
                self.joints.append(np.array(joints_list[idx], dtype=np.float32))
                self.joints_uv.append(ho3d_util.projectPoints(np.array(joints_list[idx], dtype=np.float32), K))
                self.mano_params.append(np.array(mano_list[idx], dtype=np.float32))
                self.obj_p2ds.append(np.array(obj_p2d_list[idx], dtype=np.float32))
        else:
            self.set_list = ho3d_util.load_names(os.path.join(self.root, "evaluation.txt"))

    def data_aug(self, img, mano_param, joints_uv, K, gray, p2d):
        crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)

        # Randomly jitter center
        center_offsets = (self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2))
        center = center + center_offsets

        # Scale jittering
        scale_jittering = self.scale_jittering * np.random.randn() + 1
        scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
        scale = scale * scale_jittering

        rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        affinetrans, post_rot_trans, rot_mat = dataset_util.get_affine_transform(center, scale,
                                                                              [self.inp_res, self.inp_res], rot=rot,
                                                                              K=K)
        # Change mano from openGL coordinates to normal coordinates
        mano_param[:3] = dataset_util.rotation_angle(mano_param[:3], rot_mat, coord_change_mat=self.coord_change_mat)

        joints_uv = dataset_util.transform_coords(joints_uv, affinetrans)  # hand landmark trans
        K = post_rot_trans.dot(K)

        p2d = dataset_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        #TODO: change bbox-based normalization to scaling-based normalization
        joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        #TODO: change bbox-based normalization to scaling-based normalization
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))

        # Img blurring and color jitter
        blur_radius = random.random() * self.blur_radius
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = dataset_util.color_jitter(img, brightness=self.brightness,
                                        saturation=self.saturation, hue=self.hue, contrast=self.contrast)

        # Generate object mask: gray segLabel transform and crop
        gray = dataset_util.transform_img(gray, affinetrans, [self.inp_res, self.inp_res])
        gray = gray.crop((0, 0, self.inp_res, self.inp_res))
        gray = dataset_util.get_mask_ROI(gray, bbox_obj)
        # Generate object mask
        gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        obj_mask = torch.from_numpy(obj_mask)

        return img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj

    def data_crop(self, img, K, bbox_hand, p2d):
        crop_hand = dataset_util.get_bbox_joints(bbox_hand.reshape(2, 2), bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_util.get_bbox_joints(bbox_hand.reshape(2, 2), bbox_factor=1.1)
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, _ = dataset_util.get_affine_transform(center, scale, [self.inp_res, self.inp_res])
        bbox_hand = dataset_util.transform_coords(bbox_hand.reshape(2, 2), affinetrans).flatten()
        bbox_obj = dataset_util.transform_coords(bbox_obj.reshape(2, 2), affinetrans).flatten()
        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))
        K = affinetrans.dot(K)
        return img, K, bbox_hand, bbox_obj

    def __len__(self):
        return len(self.set_list)

    def __getitem__(self, idx):
        sample = {}
        seqName, id = self.set_list[idx].split("/")
        img = ho3d_util.read_RGB_img(self.root, seqName, id, self.mode)
        if self.mode == 'train':
            K = self.K[idx]
            # hand information
            joints = self.joints[idx]
            joints_uv = self.joints_uv[idx]
            mano_param = self.mano_params[idx]
            # object information
            gray = ho3d_util.read_gray_img(self.train_seg_root, seqName, id)
            p2d = self.obj_p2ds[idx]
            # data augmentation
            sample["img_ori"] = functional.to_tensor(img)
            bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
            sample["joints_uv_ori"] = joints_uv
            #joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)
            mjm_mask, mvm_mask = dataset_util.get_vjmask(percent=0.2)
            img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj = self.data_aug(img, mano_param, joints_uv, K, gray, p2d)
            sample["img"] = functional.to_tensor(img)
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["mano_param"] = mano_param
            sample["cam_intr"] = K
            sample["joints"] = joints
            sample["joints2d"] = joints_uv
            sample["obj_p2d"] = p2d
            sample["obj_mask"] = obj_mask
            sample["mjm_mask"] = torch.from_numpy(mjm_mask)
            sample["mvm_mask"] = torch.from_numpy(mvm_mask)
        else:
            annotations = np.load(os.path.join(os.path.join(self.root, self.mode), seqName, 'meta', id + '.pkl'),
                                  allow_pickle=True)
            K = np.array(annotations['camMat'], dtype=np.float32)
            # object
            sample["obj_cls"] = annotations['objName']
            sample["obj_bbox3d"] = self.obj_bbox3d[sample["obj_cls"]]
            sample["obj_diameter"] = self.obj_diameters[sample["obj_cls"]]
            obj_pose = ho3d_util.pose_from_RT(annotations['objRot'].reshape((3,)), annotations['objTrans'])
            p2d = ho3d_util.projectPoints(sample["obj_bbox3d"], K, rt=obj_pose)
            sample["obj_pose"] = obj_pose
            # hand 
            bbox_hand = np.array(annotations['handBoundingBox'], dtype=np.float32)
            root_joint = np.array(annotations['handJoints3D'], dtype=np.float32)
            root_joint = root_joint.dot(self.coord_change_mat.T)
            sample["root_joint"] = root_joint
            img, K, bbox_hand, bbox_obj = self.data_crop(img, K, bbox_hand, p2d)
            sample["img"] = functional.to_tensor(img)
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["cam_intr"] = K
            
        return sample

class my_HO3D(HO3D):
    """
    Process the annotations extracted by my self from the pickle files in HO3D
    + gt uv coordinates of hand joints
    + gt mano translation params for hand
    + gt 3d transformed corners of object
    """
    def __init__(self, dataset_root, obj_model_root, train_label_root="./ho3d-process", mode="evaluation", inp_res=512, max_rot=np.pi, scale_jittering=0.2, center_jittering=0.1, hue=0.15, saturation=0.5, contrast=0.5, brightness=0.5, blur_radius=0.5):
        # Dataset attributes
        self.root = dataset_root
        self.mode = mode
        self.inp_res = inp_res
        self.joint_root_id = 0
        self.jointsMapManoToSimple = [0, 13, 14, 15, 16,
                                      1, 2, 3, 17,
                                      4, 5, 6, 18,
                                      10, 11, 12, 19,
                                      7, 8, 9, 20]
        self.jointsMapSimpleToMano = np.argsort(self.jointsMapManoToSimple)
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        # object informations
        self.obj_mesh = ho3d_util.load_objects_HO3D(obj_model_root)
        self.obj_bbox3d = dataset_util.get_bbox21_3d_from_dict(self.obj_mesh)
        self.obj_diameters = dataset_util.get_diameter(self.obj_mesh)
        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jittering
        self.max_rot = max_rot

        self.train_seg_root = os.path.join(train_label_root, "train_segLabel")
        if self.mode != "evaluation":
            self.mano_params = []
            self.joints = []
            self.joints_uv = []
            self.vertices = []
            self.obj_p2ds = []
            self.obj_p3ds = []
            self.obj_rots = []
            self.obj_trans = []
            self.K = []
            # training list
            total_list = ho3d_util.load_names(os.path.join(train_label_root, "train.txt"))
            #TODO: split train and val dataset
            val_seq_list = ["ABF10", "ABF11", "ABF12", "ABF13", "ABF14"]
            #val_seq_list = ["ABF10"]
            val_dict, train_dict = ho3d_util.get_train_val_split(total_list, val_seq_list)
            if self.mode == "train":
                self.set_list = train_dict['train_list']
                set_id_list = train_dict["train_id_list"]
                print(f"Train images in v3:{len(self.set_list)}; check id list: {len(set_id_list)}")
            else:
                self.set_list = val_dict['val_list']
                set_id_list = val_dict["val_id_list"]
                print(f"Validation images in v3:{len(self.set_list)}; check id list:{len(set_id_list)}")

            # camera matrix
            K_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_K.json'))
            #K_list = K_list[0]
            # hand joints
            joints_all_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_joint.json'))
            joints_3d_list = joints_all_list[0]
            joints_2d_list = joints_all_list[1]
            #vertices_3d_list = joints_all_list[2]
            # mano params
            mano_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_mano.json'))
            mano_pose_list = mano_list[0]
            mano_trans_list = mano_list[1]
            mano_beta_list = mano_list[2]
            # obj landmarks
            obj_all_list = ho3d_util.json_load(os.path.join(train_label_root, 'train_obj.json'))
            obj_p2d_list = obj_all_list[0]
            obj_p3d_list = obj_all_list[1]
            obj_rot_list = obj_all_list[2]
            obj_tran_list = obj_all_list[3]

            for i, idx in enumerate(set_id_list):
                mano_param = mano_pose_list[idx]
                mano_param.extend(mano_beta_list[idx])
                mano_param.extend(mano_trans_list[idx])
                
                self.K.append(np.array(K_list[idx], dtype=np.float32))
                self.joints.append(np.array(joints_3d_list[idx], dtype=np.float32))
                self.joints_uv.append(np.array(joints_2d_list[idx], dtype=np.float32))
                #self.vertices.append(np.array(vertices_3d_list[idx], dtype=np.float32))
                self.mano_params.append(np.array(mano_param, dtype=np.float32))
                self.obj_p2ds.append(np.array(obj_p2d_list[idx], dtype=np.float32))
                self.obj_p3ds.append(np.array(obj_p3d_list[idx], dtype=np.float32))
                self.obj_rots.append(np.array(obj_rot_list[idx], dtype=np.float32))
                self.obj_trans.append(np.array(obj_tran_list[idx], dtype=np.float32))
        else:
            self.set_list = ho3d_util.load_names(os.path.join(self.root, "evaluation.txt"))
    def data_process_val(self, joints_uv, gray, p2d):
        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)
        
        # Generate object mask
        gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        obj_mask = torch.from_numpy(obj_mask)
        return joints_uv, p2d, bbox_hand, bbox_obj, obj_mask

    def __len__(self):
        return len(self.set_list)

    def __getitem__(self, idx):
        sample = {}
        seqName, id = self.set_list[idx].split("/")
        sample['id'] = id
        sample['seq'] = seqName
        if self.mode != 'evaluation':
            img = ho3d_util.read_RGB_img(self.root, seqName, id, "train")
            K = self.K[idx]
            # hand information
            joints = self.joints[idx]
            joints = joints[self.jointsMapManoToSimple]
            joints_uv = self.joints_uv[idx]
            joints_uv = joints_uv[self.jointsMapManoToSimple]
            #joints_uv = ho3d_util.projectPoints(joints, K)
            mano_param = self.mano_params[idx]
            # object information
            gray = ho3d_util.read_gray_img(self.train_seg_root, seqName, id)
            p2d = self.obj_p2ds[idx]
            # data augmentation
            sample["img_ori"] = functional.to_tensor(img)
            bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
            sample["joints_uv_ori"] = joints_uv
            #joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)
            
            if self.mode == "train":
                img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj = self.data_aug(img, mano_param, joints_uv, K, gray, p2d)
                mjm_mask, mvm_mask = dataset_util.get_vjmask(percent=0.05) # 0.0 or 0.05 peroforms better
                sample["mjm_mask"] = torch.from_numpy(mjm_mask)
                sample["mvm_mask"] = torch.from_numpy(mvm_mask)
            else:
                annotations = np.load(os.path.join(os.path.join(self.root, "train"), seqName, 'meta', id + '.pkl'),
                                  allow_pickle=True)
                # object
                sample["obj_cls"] = annotations['objName']
                sample["obj_rot"] = annotations['objRot']
                sample["obj_tran"] = annotations['objTrans']

                joints_uv, p2d, bbox_hand, bbox_obj, obj_mask = self.data_process_val(joints_uv, gray, p2d)
            sample["img"] = functional.to_tensor(img)
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["mano_param"] = mano_param
            sample["cam_intr"] = K
            sample["joints"] = joints
            sample["joints2d"] = joints_uv
            sample["obj_p2d"] = p2d
            sample["obj_mask"] = obj_mask
            
        else:
            img = ho3d_util.read_RGB_img(self.root, seqName, id, "evaluation")
            annotations = np.load(os.path.join(os.path.join(self.root, self.mode), seqName, 'meta', id + '.pkl'),
                                  allow_pickle=True)
            K = np.array(annotations['camMat'], dtype=np.float32)
            # object
            sample["obj_cls"] = annotations['objName']
            sample["obj_bbox3d"] = self.obj_bbox3d[sample["obj_cls"]]
            sample["obj_diameter"] = self.obj_diameters[sample["obj_cls"]]
            obj_pose = ho3d_util.pose_from_RT(annotations['objRot'].reshape((3,)), annotations['objTrans'])
            p2d = ho3d_util.projectPoints(sample["obj_bbox3d"], K, rt=obj_pose)
            sample["obj_pose"] = obj_pose
            # hand 
            bbox_hand = np.array(annotations['handBoundingBox'], dtype=np.float32)
            root_joint = np.array(annotations['handJoints3D'], dtype=np.float32)
            root_joint = root_joint.dot(self.coord_change_mat.T)
            sample["root_joint"] = root_joint
            img, K, bbox_hand, bbox_obj = self.data_crop(img, K, bbox_hand, p2d)
            sample["img"] = functional.to_tensor(img)
            sample["bbox_hand"] = bbox_hand
            sample["bbox_obj"] = bbox_obj
            sample["cam_intr"] = K
            
        return sample
