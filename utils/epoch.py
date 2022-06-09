import os
from pyexpat import model
import time
from unittest import loader
import torch
import numpy as np
import cv2
import trimesh
import matplotlib
import matplotlib.pyplot as plt

from utils.utils import progress_bar as bar, AverageMeters, dump
from dataset.ho3d_util import filter_test_object, get_unseen_test_object
from utils.metric import eval_object_pose, eval_batch_obj, vertices_reprojection
from utils.visualization import *
from utils.eval_util import EvalUtil
from utils.eval_util import *
from dataset import dataset_util


def visual_in_epoch(epoch, batch_idx, visual_path_train, renderer, imgs, pred_verts, pred_camera, pred_joints2d, gt_keypoints_2d=None, gt_bbox_hand=None, idx=None, seq=None):
    print("-----Start visualization!------")
    #TODO: initialize the renderer with mano initial faces
    Mesh_visualizer = Visualizer(renderer, imgs, pred_verts, pred_camera, pred_joints2d, gt_keypoints_2d=gt_keypoints_2d, gt_bbox_hand=gt_bbox_hand, idx=idx, seq=seq)
    #print(f"check shapae of vertices:\n{pred_verts.shape}")
    print("Renderer and visualizer initialized.")
    visual_imgs = Mesh_visualizer.visualize_mesh(1, 2)
    #print(f"The size of visual_imgs: {visual_imgs.shape}")
    visual_imgs = visual_imgs.transpose(0,1)
    visual_imgs = visual_imgs.transpose(1,2)
    #print(f"The size of visual_imgs after transpose: {visual_imgs.shape}")
    visual_imgs = np.asarray(visual_imgs)

    stamp = str(epoch) + '_' + str(batch_idx)
    temp_fname = 'visual_' + stamp + '.jpg'
    temp_fname = os.path.join(visual_path_train, temp_fname)
    cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]))
    print(f"Visualization results in batch {batch_idx} saved!")


class Epoch():
    def __init__(self, dataloader, model, optimizer=None, save_path="checkpoints",mode="train", save_results=False, indices_order=None, use_cuda=False, args=None, renderer=None):
        self.loader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.mode = mode
        self.save_path = save_path
        self.save_results = save_results
        self.indices_order = indices_order
        self.use_cuda = use_cuda
        self.args = args
        self.renderer = renderer
        self.coord_change_mat = torch.tensor([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=torch.float32)

        self.time_meters = AverageMeters()
        self.visual_path_train = os.path.join(save_path, "visual_train/")
        self.visual_path_train_3d = os.path.join(save_path, "visual_train_3d/")
        self.visual_path_eval = os.path.join(save_path, "visual_eval/")
        self.visual_path_eval_3d = os.path.join(save_path, "visual_eval_3d/")
        os.makedirs(self.visual_path_train, exist_ok=True)
        os.makedirs(self.visual_path_eval, exist_ok=True)
        os.makedirs(self.visual_path_train_3d, exist_ok=True)
        os.makedirs(self.visual_path_eval_3d, exist_ok=True)
    
    def get_data(self, sample):
        imgs = sample["img"].float().cuda()
        bbox_hand = sample["bbox_hand"].float().cuda()
        bbox_obj = sample["bbox_obj"].float().cuda()
        if self.mode != "evaluation":
            mano_params = sample["mano_param"].float().cuda()
            joints_uv = sample["joints2d"].float().cuda()
            #joints = sample["joints"].float().cuda()
            obj_p2d_gt = sample["obj_p2d"].float().cuda()
            obj_mask = sample["obj_mask"].float().cuda()

            gt_camera = sample["cam_intr"].float().cuda()
            #TODO: for testing the visiualization of original annotations
            imgs_orig = sample["img_ori"].float()
            joints_uv_orig = sample["joints_uv_ori"].float()

            if self.mode == "train":
                mjm_mask = sample["mjm_mask"].float().cuda()
                mvm_mask = sample["mvm_mask"].float().cuda()
                masks = [mjm_mask, mvm_mask]
            
                return imgs, bbox_hand, bbox_obj, mano_params, joints_uv, obj_p2d_gt, obj_mask, masks, imgs_orig, joints_uv_orig, gt_camera
            else:
                return imgs, bbox_hand, bbox_obj, mano_params, joints_uv, obj_p2d_gt, obj_mask, gt_camera
        else:
            gt_camera = sample["cam_intr"].float().cuda()
            if self.use_cuda and torch.cuda.is_available():
                if "root_joint" in sample:
                    root_joints = sample["root_joint"].float().cuda()
                else:
                    root_joints = None

            else:
                gt_camera = sample["cam_intr"].float()
                if "root_joint" in sample:
                    root_joints = sample["root_joint"].float()
                else:
                    root_joints = None
            return imgs, bbox_hand, bbox_obj, gt_camera, root_joints


class Train_epoch_Trans(Epoch):
    def __init__(self, dataloader, model, optimizer=None, save_path="checkpoints", mode="train", save_results=False, indices_order=None, use_cuda=False, args=None, renderer=None):
        super().__init__(dataloader, model, optimizer, save_path, mode, save_results, indices_order, use_cuda, args, renderer)
        self.jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]


    def update(self, epoch):
        print(f"training epoch: {epoch + 1}")
        avg_meters = AverageMeters()
        self.model.train()
        end = time.time()
        for batch_idx, sample in enumerate(self.loader):
            imgs, bbox_hand, bbox_obj, mano_params, joints_uv, obj_p2d_gt, obj_mask, masks, imgs_orig, joints_uv_orig, gt_camera = self.get_data(sample)
            # measure data loading time
            self.time_meters.add_loss_value("data_time", time.time() - end)
            # model forward
            model_loss, model_losses, pred_camera, pred_3d_joints, pred_verts, pred_2d_joints, preds_obj,gt_mano_results = self.model(imgs, bbox_hand, bbox_obj, mode="train", gt_camera=gt_camera, mano_params=mano_params, mask = masks,
                                                joints_uv=joints_uv, obj_p2d_gt=obj_p2d_gt, obj_mask=obj_mask)
            #pred_verts = results["verts3d"]
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            model_loss.backward()
            self.optimizer.step()

            for key, val in model_losses.items():
                if val is not None:
                    avg_meters.add_loss_value(key, val)

            # measure elapsed time
            self.time_meters.add_loss_value("batch_time", time.time() - end)

            # plot progress
            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | lr: {lr}" \
                        "| Vertices Loss: {vertices_loss:.3f} " \
                        "| Joints3D Loss: {joints3d_loss:.3f} " \
                        "| Joints3D Mesh Loss: {joints3d_mesh_loss:.3f} | Joint2D Loss: {joints2d_loss:.3f} " \
                        "| Joint2D Mesh Loss: {joints2d_mesh_loss:.3f}  " \
                        "| Obj Reg Loss: {obj_reg_loss:.4f} | Obj conf Loss: {obj_conf_loss:.4f}" \
                        "| Total Loss: {total_loss:.3f} " \
                .format(batch=batch_idx + 1, size=len(self.loader),
                        data=self.time_meters.average_meters["data_time"].val,
                        bt=self.time_meters.average_meters["batch_time"].avg,
                        lr=self.optimizer.param_groups[0]['lr'],
                        vertices_loss=avg_meters.average_meters["vertice"].avg,
                        joints3d_loss=avg_meters.average_meters["joint3d"].avg,
                        joints3d_mesh_loss=avg_meters.average_meters["joint3d_mesh"].avg,
                        joints2d_loss=avg_meters.average_meters["joint2d"].avg,
                        joints2d_mesh_loss=avg_meters.average_meters["joint2d_mesh"].avg,
                        #hm_joints2d_loss=avg_meters.average_meters["joint2d_hm"].avg,
                        obj_reg_loss=avg_meters.average_meters["obj_reg_loss"].avg,
                        obj_conf_loss=avg_meters.average_meters["obj_conf_loss"].avg,
                        total_loss=avg_meters.average_meters["total_loss"].avg)
                    
            bar(suffix)

            # TODO: training visualization
            #if epoch % 2 == 0 and batch_idx % 2 == 0:
                #TODO: currently using original(no augmentation) image and annotations for testing the visiualization of original annotations
                # can change imgs_orig -> imgs, joints_uv_orig -> joints for training images/annot

                # TODO: change pred_verts to gt_mano_param["verts"] ; change pred_camera to gt_mano_param["mano_trans"]

                #visual_in_epoch(epoch, batch_idx, self.visual_path_train, self.renderer, imgs_orig, gt_mano_results["verts3d"]*100, pred_camera, pred_2d_joints, gt_keypoints_2d=joints_uv_orig, gt_bbox_hand=None, idx=sample['id'], seq=sample['seq'])
            end = time.time()
        return avg_meters

class Val_epoch_Trans(Epoch):
    def __init__(self, dataloader, model, optimizer=None, save_path="checkpoints", mode="train", save_results=False, indices_order=None, use_cuda=False, args=None, renderer=None):
        super().__init__(dataloader, model, optimizer, save_path, mode, save_results, indices_order, use_cuda, args, renderer)
        self.visual_path_val = os.path.join(save_path, "visual_val/")
        self.visual_path_val_3d = os.path.join(save_path, "visual_val_3d/")
        os.makedirs(self.visual_path_val, exist_ok=True)
        os.makedirs(self.visual_path_val_3d, exist_ok=True)
    
    def update(self, epoch, xyz_dict, handmesh_dict, epoch_list):
        print(f"Validation epoch: {epoch + 1}")
        avg_meters = AverageMeters()
        # TODO: 1. set eval state for self.model
        self.model.eval()
        # TODO: 2. initialize object metrics dict
        # -- REP_res_dict
        # -- ADD_res_dict
        # TODO: 3. load and initialize dict for object model
        REP_res_dict, ADD_res_dict= {}, {}
        diameter_dict = self.loader.dataset.obj_diameters
        mesh_dict = self.loader.dataset.obj_mesh
        mesh_dict, diameter_dict = filter_test_object(mesh_dict, diameter_dict)
        unseen_objects = get_unseen_test_object()
        for k in mesh_dict.keys():
            REP_res_dict[k] = []
            ADD_res_dict[k] = []

        # TODO: 4. initialize hand metrics dict
        eval_xyz, eval_xyz_procrustes_aligned, eval_xyz_sc_tr_aligned = EvalUtil(), EvalUtil(), EvalUtil()
        eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)

        # TODO: 5. call self.get_data() for validation, should return:
        # -- losses for records
        # -- imgs, bbox_hand, bbox_obj
        # -- joints_uv, mano_params

        for batch_idx, sample in enumerate(self.loader):
            end = time.time()
            imgs, bbox_hand, bbox_obj, mano_params, joints_uv, obj_p2d_gt, obj_mask, gt_camera = self.get_data(sample)

            self.time_meters.add_loss_value("data_time", time.time() - end)

            end = time.time()

            model_loss, model_losses, pred_camera, pred_3d_joints, pred_verts, pred_2d_joints, preds_obj,gt_mano_results = self.model(imgs, bbox_hand, bbox_obj, mode="validation", gt_camera=gt_camera, mano_params=mano_params, joints_uv=joints_uv, obj_p2d_gt=obj_p2d_gt, obj_mask=obj_mask)

            for key, val in model_losses.items():
                if val is not None:
                    avg_meters.add_loss_value(key, val)
            
            # TODO: 5. INFERENCE with (imgs, bbox_hand, bbox_obj), no coordinate change matrix since is not for metrics computed with annotations on the leaderboard
            # -- object metrics:
            # object predictions and evaluation(online)
            # cam_intr = sample["cam_intr"].numpy()
            # obj_pose = sample['obj_pose'].numpy()
            # obj_cls = sample['obj_cls']
            # obj_bbox3d = sample['obj_bbox3d'].numpy()
            # REP_res_dict, ADD_res_dict, reproj_pred0, reproj_gt0 = eval_batch_obj(preds_obj, bbox_obj,
            #                                             obj_pose, mesh_dict, obj_bbox3d, obj_cls,
            #                                             cam_intr, REP_res_dict, ADD_res_dict)
            # if REP_res_dict is not None and ADD_res_dict is not None \
            #     and diameter_dict is not None and unseen_objects is not None:
            #     eval_object_pose(REP_res_dict, ADD_res_dict, diameter_dict, outpath=self.save_path, unseen_objects=unseen_objects, epoch=epoch+1 if epoch is not None else None)
            
            # -- hand metrics:
            pred_joints3d = pred_3d_joints.cpu().numpy()
            pred_vertices = pred_verts.cpu().numpy()
            gt_xyz = gt_mano_results["joints3d"].cpu().numpy()
            gt_vertices = gt_mano_results["verts3d"].cpu().numpy()
            gt_root = gt_mano_results["root"].cpu().numpy()
            batch_size = gt_xyz.shape[0]
            for idx in range(batch_size):
                xyz = gt_xyz[idx, :, :3]
                verts = gt_vertices[idx, :, :]
                xyz_pred = pred_joints3d[idx, :, :3]
                verts_pred = pred_vertices[idx, :, :]
                # use eval utils to evaluate
                eval_xyz.feed(
                    xyz,
                    np.ones_like(xyz[:, 0]),
                    xyz_pred
                )

                eval_mesh_err.feed(
                    verts,
                    np.ones_like(verts[:,0]),
                    verts_pred
                )
                # scale and translation aligned predictions for xyz
                xyz_pred_sc_tr_aligned = align_sc_tr(xyz, xyz_pred)
                eval_xyz_sc_tr_aligned.feed(
                    xyz,
                    np.ones_like(xyz[:, 0]),
                    xyz_pred_sc_tr_aligned
                )

                # align predictions
                xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
                verts_pred_aligned = align_w_scale(verts, verts_pred)

                # Aligned errors
                eval_xyz_procrustes_aligned.feed(
                    xyz,
                    np.ones_like(xyz[:, 0]),
                    xyz_pred_aligned
                )
                eval_mesh_err_aligned.feed(
                    verts,
                    np.ones_like(verts[:, 0]),
                    verts_pred_aligned
                )
            
            
            # TODO: 7. post processing - 2: 2D visualization
            
            if (epoch+1) % self.args.val_visual_freq == 0 and batch_idx % self.args.val_visual_batch_interval == 0:
                visual_epoch_path = os.path.join(self.visual_path_val, "epoch_"+str(epoch+1))
                os.makedirs(visual_epoch_path, exist_ok=True)

                visual_in_epoch(epoch+1, batch_idx, visual_epoch_path, self.renderer, imgs, gt_mano_results["verts3d"]*100, pred_camera, pred_2d_joints, gt_keypoints_2d=joints_uv, gt_bbox_hand=bbox_hand, idx=sample['id'], seq=sample['seq']) #暂时不渲染verts3d
                
                
                # TODO: 8. post processing - 3: 3D visualization
                MeshVisualizer = Visualizer_3D(self.visual_path_val_3d, epoch+1, self.args)
                
                MeshVisualizer.Visualize_for_Val(gt_root, gt_vertices, pred_vertices, self.renderer.faces, sample['obj_cls'], sample['obj_rot'].numpy(), sample['obj_tran'].numpy(), sample['seq'], sample['id'], to_Open_GL=True)

            
            # TODO: print out something
            self.time_meters.add_loss_value("batch_time", time.time() - end)
            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s"\
                .format(batch=batch_idx + 1, size=len(self.loader),
                        data=self.time_meters.average_meters["data_time"].val,
                        bt=self.time_meters.average_meters["batch_time"].avg)

            bar(suffix)
            #end = time.time()

        # Calculate results
         # TODO: 6. post processing - 1: lost & acc repotr
        # -- print out in txt
        # -- print on window
        # measure elapsed time
        txt_file = self.visual_path_val + "val_metrics.txt"
        xyz_dict, mesh_dict = calculate_results(eval_xyz, eval_xyz_procrustes_aligned, eval_mesh_err, eval_mesh_err_aligned, txt_file, xyz_dict, handmesh_dict, epoch+1)

        # -- output pltfig
        fig_xyz = plotResultCurve(xyz_dict, epoch_list)
        outfig1 = self.visual_path_val + "xyz_val_fig.png"
        fig_xyz.savefig(outfig1, dpi=600)
        plt.close()

        fig_mesh = plotResultCurve(handmesh_dict, epoch_list)
        outfig2 = self.visual_path_val + "mesh_val_fig.png"
        fig_mesh.savefig(outfig2, dpi=600)
        plt.close()

        

        return avg_meters, xyz_dict, handmesh_dict
    

class Eval_epoch_Trans(Epoch):
    def __init__(self, dataloader, model, optimizer=None, save_path="checkpoints", mode="train", save_results=False, indices_order=None, use_cuda=False, args=None, renderer=None):
        super().__init__(dataloader, model, optimizer, save_path, mode, save_results, indices_order, use_cuda, args, renderer)
    
    def update(self, epoch):
        self.model.eval()
        REP_res_dict, ADD_res_dict= {}, {}
        diameter_dict = self.loader.dataset.obj_diameters
        mesh_dict = self.loader.dataset.obj_mesh
        mesh_dict, diameter_dict = filter_test_object(mesh_dict, diameter_dict)
        unseen_objects = get_unseen_test_object()
        for k in mesh_dict.keys():
            REP_res_dict[k] = []
            ADD_res_dict[k] = []

        if self.save_results:
            # save hand results for online evaluation
            xyz_pred_list, verts_pred_list = list(), list()
        end = time.time()
        for batch_idx, sample in enumerate(self.loader):
            imgs, bbox_hand, bbox_obj, gt_camera, root_joints = self.get_data(sample)
            # measure data loading time
            self.time_meters.add_loss_value("data_time", time.time() - end)

            pred_camera, pred_2d_joints, pred_3d_joints, pred_verts, preds_obj = self.model(imgs, bbox_hand, bbox_obj, mode="evaluation", roots3d=root_joints,  coord_change_mat=self.coord_change_mat)

            #pred_verts = results["verts3d"]
            
            #TODO: 2d rendered image visualization of evaluation
            #visual_in_epoch(epoch, batch_idx, visual_path_train, renderer, imgs, pred_verts, pred_camera, pred_joints2d, gt_keypoints_2d=None)

            pred_xyz = pred_3d_joints.detach().cpu().numpy()
            pred_verts = pred_verts.detach().cpu().numpy()


            if self.save_results:
                for xyz, verts in zip(pred_xyz, pred_verts):
                    if self.indices_order is not None:
                        xyz = xyz[self.indices_order]
                    xyz_pred_list.append(xyz)
                    verts_pred_list.append(verts)

            # object predictions and evaluation(online)
            cam_intr = sample["cam_intr"].numpy()
            obj_pose = sample['obj_pose'].numpy()
            obj_cls = sample['obj_cls']
            obj_bbox3d = sample['obj_bbox3d'].numpy()
            
            REP_res_dict, ADD_res_dict, reproj_pred0, reproj_gt0 = eval_batch_obj(preds_obj, bbox_obj,
                                                        obj_pose, mesh_dict, obj_bbox3d, obj_cls,
                                                        cam_intr, REP_res_dict, ADD_res_dict)
            #output the first reprojected pred mesh and gt mesh of every batch for visualization

            #TODO: visualization for 3D mesh of hand+object ; turn to a method in class 'Epoch'
            #handfile = self.visual_path_eval_3d + str(batch_idx) + "_hand" + '.ply'
            #objfile = self.visual_path_eval_3d + str(batch_idx) + "_obj" + '.ply'
            #picfile = self.visual_path_eval_3d + str(batch_idx) + "_img" + '.png'
            #pred_camera = pred_camera.detach().cpu().numpy()
            #imgs = imgs.detach().cpu().numpy()
            #print(imgs.shape)
            #img = imgs[0].transpose(1,2,0)
            #print(obj_bbox3d[0].shape)
            #bbox_hand = bbox_hand[0].detach().cpu().numpy()
            #hand_scale = max((bbox_hand[2]-bbox_hand[0]), (bbox_hand[3]-bbox_hand[1]))/2
            #hand_mesh = trimesh.Trimesh(vertices=((pred_verts[0]*3)+pred_camera[0]).squeeze(), faces=self.renderer.faces, process=False) # hand mesh需要生成
            #object_mesh = trimesh.Trimesh((reproj_pred0).squeeze(), process=False)
            #hand_mesh.export(handfile)
            #object_mesh.export(objfile)
            #cv2.imwrite(picfile, np.asarray(img[:,:,::-1]*255))

            # measure elapsed time
            self.time_meters.add_loss_value("batch_time", time.time() - end)

            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s"\
                .format(batch=batch_idx + 1, size=len(self.loader),
                        data=self.time_meters.average_meters["data_time"].val,
                        bt=self.time_meters.average_meters["batch_time"].avg)

            bar(suffix)
            end = time.time()
            
        if REP_res_dict is not None and ADD_res_dict is not None \
                and diameter_dict is not None and unseen_objects is not None:
           eval_object_pose(REP_res_dict, ADD_res_dict, diameter_dict, outpath=self.save_path, unseen_objects=unseen_objects, epoch=epoch+1 if epoch is not None else None)

        if self.save_results:
            pred_out_path = os.path.join(self.save_path, "eval_json_files/")
            os.makedirs(pred_out_path, exist_ok=True)
            pred_out_file = pred_out_path + "pred_epoch_{}.json".format(epoch+1) if epoch is not None else "pred_{}.json"
            dump(pred_out_file, xyz_pred_list, verts_pred_list)
        return None






############################################################################################################
############################################################################################################





class Train_epoch(Epoch):
    def __init__(self, dataloader, model, optimizer=None, save_path="checkpoints", train=True, save_results=False, indices_order=None, use_cuda=False, args=None, renderer=None):
        super().__init__(dataloader, model, optimizer, save_path, train, save_results, indices_order, use_cuda, args, renderer)


    def update(self, epoch):
        print(f"training epoch: {epoch + 1}")
        avg_meters = AverageMeters()
        self.model.train()
        end = time.time()
        for batch_idx, sample in enumerate(self.loader):
            imgs, bbox_hand, bbox_obj, mano_params, joints_uv, obj_p2d_gt, obj_mask, masks, imgs_orig, joints_uv_orig, gt_camera  = self.get_data(sample)
            # measure data loading time
            self.time_meters.add_loss_value("data_time", time.time() - end)
            # model forward
            model_loss, model_losses, pred_camera, pred_joints2d, results, preds_obj = self.model(imgs, bbox_hand, bbox_obj, mano_params=mano_params, mask = masks,
                                                joints_uv=joints_uv, obj_p2d_gt=obj_p2d_gt, obj_mask=obj_mask)
            pred_verts = results["verts3d"]
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            model_loss.backward()
            self.optimizer.step()

            for key, val in model_losses.items():
                if val is not None:
                    avg_meters.add_loss_value(key, val)

            # measure elapsed time
            self.time_meters.add_loss_value("batch_time", time.time() - end)

            # plot progress
            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s " \
                        "| Mano Mesh3D Loss: {mano_mesh3d_loss:.3f} " \
                        "| Mano Joints3D Loss: {mano_joints3d_loss:.3f} " \
                        "| Mano Shape Loss: {mano_shape_loss:.3f} | Mano Pose Loss: {mano_pose_loss:.3f} " \
                        "| Mano Total Loss: {mano_total_loss:.3f} | Heatmap Joints2D Loss: {hm_joints2d_loss:.3f} " \
                        "| Obj Reg Loss: {obj_reg_loss:.4f} | Obj conf Loss: {obj_conf_loss:.4f}" \
                        "| Total Loss: {total_loss:.3f} " \
                .format(batch=batch_idx + 1, size=len(self.loader),
                        data=self.time_meters.average_meters["data_time"].val,
                        bt=self.time_meters.average_meters["batch_time"].avg,
                        mano_mesh3d_loss=avg_meters.average_meters["mano_mesh3d_loss"].avg,
                        mano_joints3d_loss=avg_meters.average_meters["mano_joints3d_loss"].avg,
                        mano_shape_loss=avg_meters.average_meters["manoshape_loss"].avg,
                        mano_pose_loss=avg_meters.average_meters["manopose_loss"].avg,
                        mano_total_loss=avg_meters.average_meters["mano_total_loss"].avg,
                        hm_joints2d_loss=avg_meters.average_meters["hm_joints2d_loss"].avg,
                        obj_reg_loss=avg_meters.average_meters["obj_reg_loss"].avg,
                        obj_conf_loss=avg_meters.average_meters["obj_conf_loss"].avg,
                        total_loss=avg_meters.average_meters["total_loss"].avg)
            bar(suffix)

            # TODO: training visualization
            #if epoch % 2 == 0 and batch_idx % 2 == 0:
                #TODO: currently using original(no augmentation) image and annotations for testing the visiualization of original annotations
                # can change imgs_orig -> imgs, joints_uv_orig -> joints for training images/annot

                #visual_in_epoch(epoch, batch_idx, self.visual_path_train, self.renderer, imgs_orig, pred_verts, pred_camera, pred_joints2d, gt_keypoints_2d=joints_uv_orig)
            end = time.time()
        return avg_meters

class Eval_epoch(Epoch):
    def __init__(self, dataloader, model, optimizer=None, save_path="checkpoints", train=True, save_results=False, indices_order=None, use_cuda=False, args=None, renderer=None):
        super().__init__(dataloader, model, optimizer, save_path, train, save_results, indices_order, use_cuda, args, renderer)
    
    def update(self, epoch):
        self.model.eval()
        REP_res_dict, ADD_res_dict= {}, {}
        diameter_dict = self.loader.dataset.obj_diameters
        mesh_dict = self.loader.dataset.obj_mesh
        mesh_dict, diameter_dict = filter_test_object(mesh_dict, diameter_dict)
        unseen_objects = get_unseen_test_object()
        for k in mesh_dict.keys():
            REP_res_dict[k] = []
            ADD_res_dict[k] = []

        if self.save_results:
            # save hand results for online evaluation
            xyz_pred_list, verts_pred_list = list(), list()
        end = time.time()
        for batch_idx, sample in enumerate(self.loader):
            imgs, bbox_hand, bbox_obj, gt_camera, root_joints = self.get_data(sample)
            # measure data loading time
            self.time_meters.add_loss_value("data_time", time.time() - end)

            pred_camera, pred_joints2d, results, preds_obj = self.model(imgs, bbox_hand, bbox_obj, roots3d=root_joints)

            pred_verts = results["verts3d"]
            
            #TODO: 2d rendered image visualization of evaluation
            #visual_in_epoch(epoch, batch_idx, visual_path_train, renderer, imgs, pred_verts, pred_camera, pred_joints2d, gt_keypoints_2d=None)

            pred_xyz = results["joints3d"].detach().cpu().numpy()
            pred_verts = results["verts3d"].detach().cpu().numpy()


            if self.save_results:
                for xyz, verts in zip(pred_xyz, pred_verts):
                    if self.indices_order is not None:
                        xyz = xyz[self.indices_order]
                    xyz_pred_list.append(xyz)
                    verts_pred_list.append(verts)

            # object predictions and evaluation(online)
            cam_intr = sample["cam_intr"].numpy()
            obj_pose = sample['obj_pose'].numpy()
            obj_cls = sample['obj_cls']
            obj_bbox3d = sample['obj_bbox3d'].numpy()
            REP_res_dict, ADD_res_dict, reproj_pred0, reproj_gt0 = eval_batch_obj(preds_obj, bbox_obj,
                                                        obj_pose, mesh_dict, obj_bbox3d, obj_cls,
                                                        cam_intr, REP_res_dict, ADD_res_dict)
            #output the first reprojected pred mesh and gt mesh of every batch for visualization

            #TODO: visualization for 3D mesh of hand+object ; turn to a method in class 'Epoch'
            handfile = self.visual_path_eval_3d + str(batch_idx) + "_hand" + '.ply'
            objfile = self.visual_path_eval_3d + str(batch_idx) + "_obj" + '.ply'
            picfile = self.visual_path_eval_3d + str(batch_idx) + "_img" + '.png'
            pred_camera = pred_camera.detach().cpu().numpy()
            imgs = imgs.detach().cpu().numpy()
            #print(imgs.shape)
            img = imgs[0].transpose(1,2,0)
            #print(obj_bbox3d[0].shape)
            #bbox_hand = bbox_hand[0].detach().cpu().numpy()
            #hand_scale = max((bbox_hand[2]-bbox_hand[0]), (bbox_hand[3]-bbox_hand[1]))/2
            hand_mesh = trimesh.Trimesh(vertices=((pred_verts[0]*3)+pred_camera[0]).squeeze(), faces=self.renderer.faces, process=False) # hand mesh需要生成
            object_mesh = trimesh.Trimesh((reproj_pred0).squeeze(), process=False)
            hand_mesh.export(handfile)
            object_mesh.export(objfile)
            cv2.imwrite(picfile, np.asarray(img[:,:,::-1]*255))

            # measure elapsed time
            self.time_meters.add_loss_value("batch_time", time.time() - end)

            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s"\
                .format(batch=batch_idx + 1, size=len(self.loader),
                        data=self.time_meters.average_meters["data_time"].val,
                        bt=self.time_meters.average_meters["batch_time"].avg)

            bar(suffix)
            end = time.time()
            break
        if REP_res_dict is not None and ADD_res_dict is not None \
                and diameter_dict is not None and unseen_objects is not None:
           eval_object_pose(REP_res_dict, ADD_res_dict, diameter_dict, outpath=self.save_path, unseen_objects=unseen_objects, epoch=epoch+1 if epoch is not None else None)

        if self.save_results:
            pred_out_path = os.path.join(self.save_path, "pred_epoch_{}.json".format(epoch+1) if epoch is not None else "pred_{}.json")
            dump(pred_out_path, xyz_pred_list, verts_pred_list)
        return None



    

def single_epoch(loader, model, epoch=None, optimizer=None, save_path="checkpoints",
                 train=True, save_results=False, indices_order=None, use_cuda=False, args=None, renderer=None):

    time_meters = AverageMeters()
    visual_path_train = os.path.join(save_path, "visual_train/")
    visual_path_train_3d = os.path.join(save_path, "visual_train_3d/")
    visual_path_eval = os.path.join(save_path, "visual_eval/")
    visual_path_eval_3d = os.path.join(save_path, "visual_eval_3d/")
    os.makedirs(visual_path_train, exist_ok=True)
    os.makedirs(visual_path_eval, exist_ok=True)
    os.makedirs(visual_path_train_3d, exist_ok=True)
    os.makedirs(visual_path_eval_3d, exist_ok=True)

    if train:
        print(f"training epoch: {epoch + 1}")
        avg_meters = AverageMeters()
        model.train()

    else:
        model.eval()

        # object evaluation
        REP_res_dict, ADD_res_dict= {}, {}
        diameter_dict = loader.dataset.obj_diameters
        mesh_dict = loader.dataset.obj_mesh
        mesh_dict, diameter_dict = filter_test_object(mesh_dict, diameter_dict)
        unseen_objects = get_unseen_test_object()
        for k in mesh_dict.keys():
            REP_res_dict[k] = []
            ADD_res_dict[k] = []

        if save_results:
            # save hand results for online evaluation
            xyz_pred_list, verts_pred_list = list(), list()

    end = time.time()
    for batch_idx, sample in enumerate(loader):
        if train:
            assert use_cuda and torch.cuda.is_available(), "requires cuda for training"
            imgs = sample["img"].float().cuda()
            bbox_hand = sample["bbox_hand"].float().cuda()
            bbox_obj = sample["bbox_obj"].float().cuda()

            mano_params = sample["mano_param"].float().cuda()
            joints_uv = sample["joints2d"].float().cuda()
            obj_p2d_gt = sample["obj_p2d"].float().cuda()
            obj_mask = sample["obj_mask"].float().cuda()

            #gt_camera = sample["cam_intr"].float().cuda()
            #TODO: for testing the visiualization of original annotations
            imgs_orig = sample["img_ori"].float()
            joints_uv_orig = sample["joints_uv_ori"].float()

            # measure data loading time
            time_meters.add_loss_value("data_time", time.time() - end)
            # model forward
            model_loss, model_losses, pred_camera, pred_joints2d, results, preds_obj = model(imgs, bbox_hand, bbox_obj, mano_params=mano_params,
                                             joints_uv=joints_uv, obj_p2d_gt=obj_p2d_gt, obj_mask=obj_mask)
            pred_verts = results["verts3d"]
            # compute gradient and do SGD step
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            for key, val in model_losses.items():
                if val is not None:
                    avg_meters.add_loss_value(key, val)

            # measure elapsed time
            time_meters.add_loss_value("batch_time", time.time() - end)

            # plot progress
            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s " \
                     "| Mano Mesh3D Loss: {mano_mesh3d_loss:.3f} " \
                     "| Mano Joints3D Loss: {mano_joints3d_loss:.3f} " \
                     "| Mano Shape Loss: {mano_shape_loss:.3f} | Mano Pose Loss: {mano_pose_loss:.3f} " \
                     "| Mano Total Loss: {mano_total_loss:.3f} | Heatmap Joints2D Loss: {hm_joints2d_loss:.3f} " \
                     "| Obj Reg Loss: {obj_reg_loss:.4f} | Obj conf Loss: {obj_conf_loss:.4f}" \
                     "| Total Loss: {total_loss:.3f} " \
                .format(batch=batch_idx + 1, size=len(loader),
                        data=time_meters.average_meters["data_time"].val,
                        bt=time_meters.average_meters["batch_time"].avg,
                        mano_mesh3d_loss=avg_meters.average_meters["mano_mesh3d_loss"].avg,
                        mano_joints3d_loss=avg_meters.average_meters["mano_joints3d_loss"].avg,
                        mano_shape_loss=avg_meters.average_meters["manoshape_loss"].avg,
                        mano_pose_loss=avg_meters.average_meters["manopose_loss"].avg,
                        mano_total_loss=avg_meters.average_meters["mano_total_loss"].avg,
                        hm_joints2d_loss=avg_meters.average_meters["hm_joints2d_loss"].avg,
                        obj_reg_loss=avg_meters.average_meters["obj_reg_loss"].avg,
                        obj_conf_loss=avg_meters.average_meters["obj_conf_loss"].avg,
                        total_loss=avg_meters.average_meters["total_loss"].avg)
            bar(suffix)

            # TODO: training visualization
            #if epoch % 2 == 0 and batch_idx % 2 == 0:
                #TODO: currently using original(no augmentation) image and annotations for testing the visiualization of original annotations
                # can change imgs_orig -> imgs, joints_uv_orig -> joints for training images/annot

                #visual_in_epoch(epoch, batch_idx, visual_path_train, renderer, imgs_orig, pred_verts, pred_camera, pred_joints2d, gt_keypoints_2d=joints_uv_orig)
            end = time.time()

        else:
            if use_cuda and torch.cuda.is_available():
                imgs = sample["img"].float().cuda()
                bbox_hand = sample["bbox_hand"].float().cuda()
                bbox_obj = sample["bbox_obj"].float().cuda()
                
                gt_camera = sample["cam_intr"].float().cuda()
                if "root_joint" in sample:
                    root_joints = sample["root_joint"].float().cuda()
                else:
                    root_joints = None

            else:
                imgs = sample["img"].float()
                bbox_hand = sample["bbox_hand"].float()
                bbox_obj = sample["bbox_obj"].float()
                gt_camera = sample["cam_intr"].float()
                if "root_joint" in sample:
                    root_joints = sample["root_joint"].float()
                else:
                    root_joints = None
            

            # measure data loading time
            time_meters.add_loss_value("data_time", time.time() - end)

            pred_camera, pred_joints2d, results, preds_obj = model(imgs, bbox_hand, bbox_obj, roots3d=root_joints)

            pred_verts = results["verts3d"]
            
            #TODO: 2d rendered image visualization of evaluation
            #visual_in_epoch(epoch, batch_idx, visual_path_train, renderer, imgs, pred_verts, pred_camera, pred_joints2d, gt_keypoints_2d=None)

            pred_xyz = results["joints3d"].detach().cpu().numpy()
            pred_verts = results["verts3d"].detach().cpu().numpy()


            if save_results:
                for xyz, verts in zip(pred_xyz, pred_verts):
                    if indices_order is not None:
                        xyz = xyz[indices_order]
                    xyz_pred_list.append(xyz)
                    verts_pred_list.append(verts)

            # object predictions and evaluation(online)
            cam_intr = sample["cam_intr"].numpy()
            obj_pose = sample['obj_pose'].numpy()
            obj_cls = sample['obj_cls']
            obj_bbox3d = sample['obj_bbox3d'].numpy()
            REP_res_dict, ADD_res_dict, reproj_pred0, reproj_gt0 = eval_batch_obj(preds_obj, bbox_obj,
                                                        obj_pose, mesh_dict, obj_bbox3d, obj_cls,
                                                        cam_intr, REP_res_dict, ADD_res_dict)
            #output the first reprojected pred mesh and gt mesh of every batch for visualization

            #TODO: visualization for 3D mesh of hand+object
            handfile = visual_path_eval_3d + str(batch_idx) + "_hand" + '.ply'
            objfile = visual_path_eval_3d + str(batch_idx) + "_obj" + '.ply'
            picfile = visual_path_eval_3d + str(batch_idx) + "_img" + '.png'
            pred_camera = pred_camera.detach().cpu().numpy()
            imgs = imgs.detach().cpu().numpy()
            #print(imgs.shape)
            img = imgs[0].transpose(1,2,0)
            #print(obj_bbox3d[0].shape)
            #bbox_hand = bbox_hand[0].detach().cpu().numpy()
            #hand_scale = max((bbox_hand[2]-bbox_hand[0]), (bbox_hand[3]-bbox_hand[1]))/2
            hand_mesh = trimesh.Trimesh(vertices=((pred_verts[0]*3)+pred_camera[0]).squeeze(), faces=renderer.faces, process=False) # hand mesh需要生成
            object_mesh = trimesh.Trimesh((reproj_pred0).squeeze(), process=False)
            hand_mesh.export(handfile)
            object_mesh.export(objfile)
            cv2.imwrite(picfile, np.asarray(img[:,:,::-1]*255))

            # measure elapsed time
            time_meters.add_loss_value("batch_time", time.time() - end)

            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s"\
                .format(batch=batch_idx + 1, size=len(loader),
                        data=time_meters.average_meters["data_time"].val,
                        bt=time_meters.average_meters["batch_time"].avg)

            bar(suffix)
            end = time.time()
            break

    if train:
        return avg_meters
    else:
        # object pose evaluation
        if REP_res_dict is not None and ADD_res_dict is not None \
                and diameter_dict is not None and unseen_objects is not None:
           eval_object_pose(REP_res_dict, ADD_res_dict, diameter_dict, outpath=save_path, unseen_objects=unseen_objects,
                            epoch=epoch+1 if epoch is not None else None)

        if save_results:
            pred_out_path = os.path.join(save_path, "pred_epoch_{}.json".format(epoch+1) if epoch is not None else "pred_{}.json")
            dump(pred_out_path, xyz_pred_list, verts_pred_list)
        return None