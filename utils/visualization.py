import sys
import math
import torch
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
from torchvision.utils import make_grid
from dataset.dataset_util import recover_joints
from utils.vis_utils import *


def plotResultCurve(dict, epoch):
    #auc = dict1["auc"]
    #mean = dict1["mean"]
    auc_al = dict["al_auc"]
    mean_al = dict["al_mean"]
    fig = plt.figure(figsize=(6,4))
    figManager = plt.get_current_fig_manager()
    
    ax0 = fig.add_subplot(2,1,1)
    #ax0.plot(epoch, auc, "r")
    ax0.plot(epoch, auc_al, "b")
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    ax0.title.set_text("Aligned Metrics Display (Validation)")

    ax1 = fig.add_subplot(2,1,2)
    #ax1.plot(epoch, mean, "r")
    ax1.plot(epoch, mean_al, "g")
    plt.xlabel("epoch")
    plt.ylabel("mean_3d_kp_error")
    #ax1.title.set_text("Mean 3D Keypoint Error (Validation)")

    return fig

def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 21. but if not will transpose it.
    """

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': (197, 27, 125),  # L lower leg
        'light_pink': (233, 163, 201),  # L upper leg
        'light_green': (161, 215, 106),  # L lower arm
        'green': (77, 146, 33),  # L upper arm
        'red': (215, 48, 39),  # head
        'light_red': (252, 146, 114),  # head
        'light_orange': (252, 141, 89),  # chest
        'purple': (118, 42, 131),  # R lower leg
        'light_purple': (175, 141, 195),  # R upper
        'light_blue': (145, 191, 219),  # R lower arm
        'blue': (69, 117, 180),  # R upper arm
        'gray': (130, 130, 130),  #
        'white': (255, 255, 255),  #
    }

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)
    
    #print(f"the 0-dim of joints:{joints.shape[0]}. the 1-dim of joints:{joints.shape[1]}.")

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    
    if joints.shape[1] == 21:  # hand
        parents = np.array([
            -1,
            0,1,2,3,
            0,5,6,7,
            0,9,10,11,
            0,13,14,15,
            0,17,18,19,
        ])
        parents_0 = np.array([
            -1,
            0,1,2,
            0,4,5,
            0,7,8,
            0,10,11,
            0,13,14,
            15,3,6,12,9,
        ])
        ecolors = {
            0: 'light_purple',
            1: 'light_green',
            2: 'light_green',
            3: 'light_green',
            4: 'pink',
            5: 'pink',
            6: 'pink',
            7: 'light_blue',
            8: 'light_blue',
            9: 'light_blue',
            10: 'light_red',
            11: 'light_red',
            12: 'light_red',
            13: 'purple',
            14: 'purple',
            15: 'purple',
            16: 'purple',
            17: 'light_green',
            18: 'pink',
            19: 'light_red',
            20: 'light_blue',
        }
    else:
        print('Unknown skeleton!!')

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                       -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                import ipdb
                ipdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image

def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    black = (255, 255, 0)
    margin = 15
    start_x = 5
    start_y = margin
    for key in sorted(content.keys()):
        text = "%s: %.2g" % (key, content[key])
        cv2.putText(image, text, (start_x, start_y), 0, 0.45, black)
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image

def showHandJoints(imgInOrg, gtIn, filename=None):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param filename: dump image name
    :return:
    '''
    import cv2

    imgIn = np.copy(imgInOrg)

    # Set color for each finger
    joint_color_code = [[139, 53, 255], # root
                        [0, 56, 255], # thumb
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [255, 255, 0]] # pinky

    # 统一的limbs链接顺序：0 root joint， 1 - 4 thumb, 5 - 8, 9 - 12, 13 - 16, 17 - 20 pinky
    # 因此输入前需要根据dataset具体的map将joint order重新排列
    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = sys.version_info[0]

    gtIn = np.round(gtIn).astype(np.int)

    if gtIn.shape[0]==1:
        imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0],
                             thickness=-1)
    else:

        for joint_num in range(gtIn.shape[0]):

            color_code_num = (joint_num // 4)
            # 圈画各个joint
            if joint_num in [0, 4, 8, 12, 16, 20]: 
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=5)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

        for limb_num in range(len(limbs)):

            x1 = gtIn[limbs[limb_num][0], 1] # hand bone端点1横坐标，2D坐标中图像中某点横坐标对应axis=1处取值
            y1 = gtIn[limbs[limb_num][0], 0] # hand bone端点1纵坐标，2D坐标中图像中某点纵坐标对应axis=0处取值
            x2 = gtIn[limbs[limb_num][1], 1] # hand bone端点2横坐标
            y2 = gtIn[limbs[limb_num][1], 0] # hand bone端点2纵坐标
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < 150 and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3), # line_width = 3
                                           int(deg), # 两点连线与纵坐标轴夹角(in degrees not radius)
                                           0, 360, 1) 

                # determine color for each limb
                color_code_num = limb_num // 4
                if PYTHON_VERSION == 3:
                    limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
                else:
                    limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]) # 通过iterator在每根手指上可画出渐变色， 35是渐变程度unit

                cv2.fillConvexPoly(imgIn, polygon, color=limb_color) #描线


    if filename is not None:
        cv2.imwrite(filename, imgIn)

    return imgIn

def showObjJoints(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    '''
    Utility function for displaying object annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''
    import cv2
    jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]
    jointColsGt = (238,238,175)
    newCol = (jointColsGt[0] + jointColsGt[1] + jointColsGt[2]) / 3
    jointColsEst  = (newCol, newCol, newCol)

    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j+1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC,0]), int(gt[jntC,1])), (int(gt[jntN,0]), int(gt[jntN,1])), jointColsGt, lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst, lineThickness)
    if gtIn.shape[0] > 8:
        gtIn = np.round(gtIn).astype(np.int)
        colorspace = [[139,0,139],
                      [255,0,0],
                      [114,128,250],
                      [0,255,255]]
        for i in range(8, gtIn.shape[0]):
            coloridx = ((i - 8) // 4)
            if gtIn is not None:
                cv2.circle(img, center=(gtIn[i][0], gtIn[i][1]), radius=2, color=colorspace[coloridx], thickness=3)
            if estIn is not None:
                cv2.circle(img, center=(gtIn[i][0], gtIn[i][1]), radius=2, color=colorspace[coloridx], thickness=3)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img

class Visualizer(object):
    def __init__(self, renderer, images, pred_vertices, pred_camera, pred_keypoints_2d=None, gt_keypoints_2d=None, gt_bbox_hand = None, score=None, is_train=True, idx=None, seq=None):
        self.renderer = renderer
        self.images = images
        self.pred_vertices = pred_vertices
        self.pred_camera = pred_camera
        self.pred_keypoints_2d = pred_keypoints_2d
        self.gt_keypoints_2d = gt_keypoints_2d
        self.gt_bbox_hand = gt_bbox_hand
        self.score = score
        self.is_train = is_train
        self.to_lsp = list(range(21)) # all the 21 joints label order consistent with the order in dataset
        self.rend_imgs = []
        self.batch_size = self.pred_vertices.shape[0]
        self.id = idx
        self.seq = seq

    def get_one_sample(self, i):
        img = self.images[i].detach().cpu().numpy().transpose(1,2,0)
        vertices = self.pred_vertices[i].detach().cpu().numpy()
        vertices = vertices.transpose(1,0)
        #print(f"check shape of vertices: {vertices.shape}")
        cam = self.pred_camera[i].detach().cpu().numpy()
        #print(f"check shape of pred_camera: {cam.shape}")
        pred_keypoints_2d = self.pred_keypoints_2d[i].detach().cpu().numpy()
        if self.gt_bbox_hand is not None:
            gt_bbox_hand = self.gt_bbox_hand[i].detach().cpu().numpy()
            pred_keypoints_2d = recover_joints(pred_keypoints_2d, gt_bbox_hand)
        #pred_keypoints_2d = pred_keypoints_2d.transpose(1,0)
        
        #print(f"check pred_kp_2d shape: {pred_keypoints_2d.shape}")
        #pred_keypoints_2d = self.pred_keypoints_2d.numpy()[i, self.to_lsp] 
        # here the pred_keypoints_2d loaded is list, so no .cpu()
        if self.gt_keypoints_2d is not None:
            gt_keypoints_2d = self.gt_keypoints_2d[i].detach().cpu().numpy()[:,:2]
            if self.gt_bbox_hand is not None:
                gt_keypoints_2d = recover_joints(gt_keypoints_2d, gt_bbox_hand)
            #gt_keypoints_2d = gt_keypoints_2d.transpose(1,0)
            #print(f"check gt_kp_2d shape: {gt_keypoints_2d.shape}")
            return img, gt_keypoints_2d, pred_keypoints_2d, vertices, cam
        else:
            return img, pred_keypoints_2d, vertices, cam


    def visualization(self, i, img_size,  color='pink', focal_length=1000, eps = 1e-9):
        if self.gt_keypoints_2d is not None:
            img, gt_kp, pred_kp, vertices, camera = self.get_one_sample(i)
            # gt_vis是控制gt的
            gt_vis = gt_kp[1,:].astype(bool)
        else:
            img, pred_kp, vertices, camera = self.get_one_sample(i)
        #loss = np.sum((gt_kp[gt_vis, :2] - pred_kp[gt_vis])**2)
        #debug_text = {"sc": camera[0], "tx": camera[1], "ty": camera[2], "kpl": loss}
        #print(f"the shape of camera matrix:\n {camera}")
        debug_text = {"sc": camera[0], "tx": camera[1], "ty": camera[2]}
        res = img.shape[1]
        camera_t = np.array([camera[1], camera[2], 2*focal_length/(res * camera[0] + eps)])
        #print(f"the shape of camera transpose matrix:\n {camera_t}")
        rend_img = self.renderer.render(vertices, camera_t=camera_t,
                                        img=img, use_bg=True,
                                        focal_length=focal_length,
                                        body_color=color)
        rend_img = draw_text(rend_img, debug_text)
        # Draw skeleton
        #pred_joint = ((pred_kp + 1) * 0.5) * img_size
        pred_joint = pred_kp
        #gt_joint = gt_kp
        #pred_joint = pred_kp
        #img_with_gt = draw_skeleton(img, gt_joint, draw_edges=False, vis=gt_vis)
        if self.gt_keypoints_2d is not None:
            #print(f"check gt_joint value: \n {gt_kp}")
            #gt_joint = ((gt_kp[:2, :] + 1) * 0.5) * img_size
            #gt_joint = gt_kp[:, :2]
            img_cv = cv2.cvtColor(img*255.0,cv2.COLOR_RGB2BGR)
            #img_with_gt = draw_skeleton(img, gt_joint, vis=gt_vis)
            img_with_gt = showHandJoints(img_cv, gt_kp)
            img_with_gt = cv2.cvtColor(img_with_gt, cv2.COLOR_BGR2RGB)
            skel_img = img_with_gt
            
        else:
            skel_img = img
        #rend_img = draw_skeleton(rend_img*255.0, pred_joint)
       #rend_img = draw_skeleton(img*255.0, pred_joint) # TODO: 暂时不做render
        rend_img = showHandJoints(img_cv, pred_joint)
        rend_img = cv2.cvtColor(rend_img, cv2.COLOR_BGR2RGB)

        combined = np.hstack([skel_img, rend_img])
        
        return combined

    def visualize_mesh(self, vis_num, nrow):
        for i in range(min(self.batch_size, vis_num)):
            rend_img = self.visualization(i, img_size=224)
            rend_img = rend_img.transpose(2,0,1)
            self.rend_imgs.append(torch.from_numpy(rend_img))
            print(f"Visualizing {self.seq[i]}_{self.id[i]}.")
        self.rend_imgs = make_grid(self.rend_imgs, nrow=nrow)
        return self.rend_imgs


class Visualizer_3D(object):
    def __init__(self, output_root, epoch, args):
        self.obj_mesh_path = args.obj_model_root
        self.output_root = output_root + "epoch_"+str(epoch) +'/'
        os.makedirs(self.output_root, exist_ok=True)
        self.center_idx = args.root_idx
        self.coord_change_mat = torch.tensor([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=torch.float32)
    
    def get_mesh_format(self, vertices, faces, color):
        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(np.copy(faces))
        mesh.vertices = o3d.utility.Vector3dVector(np.copy(vertices))
        numVert = vertices.shape[0]
        if color == 'r':
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.6, 0.2, 0.2]]), [numVert, 1]))
        elif color == 'g':
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))
        elif isinstance(color,np.ndarray):
            assert color.shape == np.array(mesh.vertices).shape
            mesh.vertex_colors = o3d.utility.Vector3dVector(color)
        else:
            raise Exception('Unknown mesh color')
        
        return mesh
        
    def visualize_one_sample(self, hand_vertices, hand_faces, obj_cls, objRot, objTrans, seq, id, mode="gt"):
        objMesh = read_obj(os.path.join(self.obj_mesh_path, obj_cls, 'textured_simple.obj'))
        objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(objRot)[0].T) + objTrans

        o3dMeshList = [self.get_mesh_format(hand_vertices, hand_faces, 'r'), self.get_mesh_format(objMesh.v, objMesh.f, 'g')]
        o3d.io.write_triangle_mesh(self.output_root + seq+'_' + id +'_hand_'+ mode + '.ply', o3dMeshList[0])
        o3d.io.write_triangle_mesh(self.output_root + seq +'_'+ id +'_obj_'+ mode + '.ply', o3dMeshList[1])

        return None
    
    def Visualize_for_Val(self, gt_roots, gt_vertices, pred_vertices, hand_faces, obj_cls, objRot, objTrans, seq, id, to_Open_GL=True):
        #gt_root = gt_roots[0]
        # if to_Open_GL:
        #     gt_root = gt_root.dot(self.coord_change_mat.T)
        #     gt_vertices = gt_vertices[0] + gt_root
        #     pred_vertices = pred_vertices[0] + gt_root
        #     gt_vertices = gt_vertices.dot(self.coord_change_mat.T)
        #     pred_vertices = pred_vertices.dot(self.coord_change_mat.T)
        # else:
        #     gt_vertices = gt_vertices[0] + gt_root
        #     pred_vertices = pred_vertices[0] + gt_root
        gt_vertices = gt_vertices[0]
        pred_vertices = pred_vertices[0]
        
        obj_cls = obj_cls[0]
        objRot = objRot[0]
        objTrans = objTrans[0]
        seq = seq[0]
        id = id[0]
        self.visualize_one_sample(gt_vertices, hand_faces, obj_cls, objRot, objTrans, seq, id)
        print(f"3d visualization of gt {seq}_{id} is saved.")
        self.visualize_one_sample(pred_vertices, hand_faces, obj_cls, objRot, objTrans, seq, id, mode="pred")
        print(f"3d visualization of pred {seq}_{id} is saved.")
        return None
