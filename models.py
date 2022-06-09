import torch
from torch import nn
from torchvision import ops
from networks.backbone import FPN
#from networks.hand_head import hand_Encoder, hand_regHead
from networks.my_hand_head import hand_Encoder, hand_regHead, hand_feature_Extractor, hand_trans_Encoder
from networks.object_head import obj_regHead, Pose2DLayer
import networks.mano_head as mano_head
from networks.mano_head import batch_rodrigues,mano_regHead
from networks.CR import Transformer
from networks.loss import Joint2DLoss, ManoLoss, ObjectLoss
import networks.loss as Loss
from src.utils.geometric_layers import orthographic_projection, K_projection
import src.modeling.data.config as cfg
from src.modeling._mano import MANO, Mesh
from dataset.ho3d_util import projectPoints
#from assets.mano_models.webuser.smpl_handpca_wrapper_HAND_only import load_model

#MANO_MODEL_PATH = "./assets/mano_models/MANO_RIGHT.pkl"

class HONet_Trans(nn.Module):
    def __init__(self, args, roi_res=32, joint_nb=21, stacks=1, channels=256, blocks=1,
                 transformer_depth=1, transformer_head=8,
                 mano_layer=None, mano_neurons=[1024, 512], coord_change_mat=None,
                 reg_object=True, pretrained=True):

        super(HONet_Trans, self).__init__()

        self.out_res = roi_res

        self.pose6d_size = 16 * 6
        self.mano_pose_size = 16 * 3

        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)

        # hand head
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb,
                                      stacks=stacks, channels=channels, blocks=blocks)
        #TODO: initialize hand_pre_encoder
        self.hand_pre_encoder = hand_feature_Extractor(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        #TODO: transformer encoder: self.hand_trans_encoder = hand_trans_Encoder()
        self.hand_trans_encoder = hand_trans_Encoder(args=args)
        self.upsampling = torch.nn.Linear(195, 778)
        # hand encoder
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        # mano branch
        #self.mano_branch = mano_regHead(mano_layer, feature_size=self.hand_encoder.#num_feat_out,
        #                                mano_neurons=mano_neurons,coord_change_mat=coord_change_mat)
        self.mano_layer = mano_layer
        # object head
        self.reg_object = reg_object
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels//2, joint_nb=joint_nb)
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)

        # CR blocks
        self.transformer = Transformer(inp_res=roi_res, dim=channels,
                                       depth=transformer_depth, num_heads=transformer_head)
        #TODO: add hand template to features
        template_pose = torch.zeros((1,48)).cuda()
        template_betas = torch.zeros((1,10)).cuda()
        template_trans = torch.zeros((1,3)).cuda()
        mano_layer = mano_layer.cuda()
        template_vertices, template_3d_joints, _ = mano_layer(template_pose, template_betas, template_trans)
        # here use the mano_layer revised for ho3d (may reach better result)
        template_vertices = template_vertices/1000.0
        template_3d_joints = template_3d_joints/1000.0
        
        self.mesh_sampler = Mesh()
        template_vertices_sub = self.mesh_sampler.downsample(template_vertices)

        #TODO: normalize joints and vertices
        template_root = template_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
        
        self.template_3d_joints = template_3d_joints - template_root[:, None, :]
        self.template_vertices = template_vertices - template_root[:, None, :]
        self.template_vertices_sub = template_vertices_sub - template_root[:, None, :]
        self.num_joints = template_3d_joints.shape[1]

        #if coord_change_mat is not None:
        #    self.register_buffer("coord_change_mat", coord_change_mat)
        #else:
        #    self.coord_change_mat = None
        


    def net_forward(self, imgs, bbox_hand, bbox_obj, mano_params=None, roots3d=None,mask=None, is_train=False):
        batch = imgs.shape[0]
        device = imgs.device
        inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
        inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
        bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1)
        msk_inter = ((inter_bottomRight-inter_topLeft > 0).sum(dim=1)) == 2
        
        #TODO: extract gt vertices and joints
        gt_mano_results = {}
        if mano_params is not None:
            mano_shape_size = 10
            gt_mano_pose = mano_params[:, :self.mano_pose_size].contiguous()
            gt_mano_shape = mano_params[:, self.mano_pose_size:(self.mano_pose_size+mano_shape_size)]
            gt_mano_trans = mano_params[:, (self.mano_pose_size+mano_shape_size):]
            gt_mano_pose_rotmat = mano_head.batch_rodrigues(gt_mano_pose.view(-1, 3)).view(-1, 16, 3, 3)
            gt_verts, gt_joints, center_joint = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape, th_trans=gt_mano_trans)
            #gt_vertices = gt_verts/1000.0
            gt_vertices = gt_verts
            #print(gt_vertices[0])
            #gt_3d_joints  = gt_joints/1000.0
            gt_3d_joints = gt_joints

            #device0 = self.mesh_sampler.device
            gt_vertices_sub = self.mesh_sampler.downsample(gt_vertices)
            
            # normalize gt based on hand's wrist 
            gt_3d_root = gt_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
            gt_vertices = gt_vertices - gt_3d_root[:, None, :]
            gt_vertices_sub = gt_vertices_sub - gt_3d_root[:, None, :]
            gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
            gt_3d_joints_with_tag = torch.ones((batch,gt_3d_joints.shape[1],4)).cuda() # TODO: the tag refers to whether annotations exist or not
            gt_3d_joints_with_tag[:,:,:3] = gt_3d_joints

            gt_mano_results = {
                "verts3d": gt_vertices,
                "verts3d_down":gt_vertices_sub,
                "joints3d": gt_3d_joints_with_tag,
                "mano_shape": gt_mano_shape,
                "mano_trans":gt_mano_trans,
                "mano_pose": gt_mano_pose_rotmat,
                "root": center_joint}
        # P2 from FPN Network
        P2 = self.base_net(imgs)[0]
        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        # get roi boxes
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        # 4 here is the downscale size in FPN network(P2)
        x = ops.roi_align(P2, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand
        #print(f"check hand roi shape: {x.shape}")
        
        #print(f"")
        # hand forward
        out_hm, encoding, hm_2d_joints = self.hand_head(x)
        #print(f"check heatmap shape: {out_hm[-1].shape}; check encoding vector shape: {encoding[-1].shape}")
        #TODO:transformer branch - forward hand_pre_encoder for image_feat and grid_feat
        image_feat, grid_feat = self.hand_pre_encoder(out_hm, encoding)
        
        # transfer the template data to current device
        self.template_3d_joints = self.template_3d_joints.to(device)
        self.template_vertices_sub = self.template_vertices_sub.to(device)
        
        ref_vertices = torch.cat([self.template_3d_joints, self.template_vertices_sub],dim=1)
        ref_vertices = ref_vertices.expand(batch, -1, -1)
        ref_joints = self.template_3d_joints.expand(batch, -1, -1)

        image_feat = image_feat.view(batch, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        #print(f"check shape of image_feat:{image_feat.shape}")
        #TODO: concatinate image feat and template mesh to form the joint/vertex queries
        features = torch.cat([ref_vertices.to(image_feat.device), image_feat], dim=2)

        #TODO: concatenate template joints to grid_features to form 'joint_grid_feat'
        joint_grid_feat = torch.cat([ref_joints.to(grid_feat.device), grid_feat], dim=2) 
        #print(f"check shape of grid_features:{joint_grid_feat.shape}")
        #TODO: add the joint_grid_feat to the regular joint feature
        features[:,:21,:] = features[:,:21,:] + joint_grid_feat
        #print(f"check shape of features:{features.shape}") # should be: B x 799 x 3
        
        if is_train == True and mask is not None:
            mjm_mask = mask[0]
            mvm_mask = mask[1]
            special_tokens = torch.ones_like(features).cuda()*0.01
            features[:,:21,:] = features[:,:21,:]*mjm_mask + special_tokens[:,:21,:]*(1 - mjm_mask)
            #features[:,-21:,:] = features[:,-21:,:]*mjm_mask + special_tokens[:,-21:,:]*(1 - mjm_mask)
            features[:,21:,:] = features[:,21:,:]*mvm_mask + special_tokens[:,21:,:]*(1 - mvm_mask)

        ## mano branch
        #mano_encoding = self.hand_encoder(out_hm, encoding)
        ## print(f"check mano_encoding shape: {mano_encoding.shape}")

        #pred_camera, pred_mano_results, gt_mano_results = self.mano_branch(mano_encoding, mano_params=mano_params, roots3d=roots3d)
        ## print(f"check shape of pred_camera: \n {pred_camera.shape}")
        #TODO: forward the hand_trans_encoder
        att_output, pred_3d_joints, pred_vertices_sub, pred_camera = self.hand_trans_encoder(features)

        #TODO: Upsample the predicted sub vertices
        temp_transpose = pred_vertices_sub.transpose(1,2)
        pred_vertices = self.upsampling(temp_transpose)
        pred_vertices = pred_vertices.transpose(1,2)
        pred_vertices = [pred_vertices, pred_vertices_sub]

        # obj forward
        if self.reg_object:
            roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)
            roi_boxes_inter = torch.cat((idx_tensor, bbox_inter), dim=1)

            y = ops.roi_align(P2, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # obj

            z = ops.roi_align(P2, roi_boxes_inter, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # intersection
            z = msk_inter[:, None, None, None] * z
            y = self.transformer(y, z)
            out_fm = self.obj_head(y)
            preds_obj = self.obj_reorgLayer(out_fm)
        else:
            preds_obj = None
        
        return att_output, hm_2d_joints, pred_3d_joints, pred_vertices, pred_camera, preds_obj, gt_mano_results

    def forward(self, imgs, bbox_hand, bbox_obj, mano_params=None, mask=None, roots3d=None, coord_change_mat=None, mode="train"):
        if mode!="evaluation": # 'training' is the attribute of nn.Module which can be configured by model.train() and model.eval()
            att_output, hm_2d_joints, pred_3d_joints, pred_vertices, pred_camera, preds_obj, gt_mano_results = self.net_forward(imgs, bbox_hand, bbox_obj, mano_params=mano_params, mask=mask, is_train=True)
            
        else:
            att_output, hm_2d_joints, pred_3d_joints, pred_vertices, pred_camera, preds_obj, gt_mano_results = self.net_forward(imgs, bbox_hand, bbox_obj, mano_params=None, is_train=False)
            if roots3d is not None: # evaluation
                roots3d = roots3d.unsqueeze(dim=1)
                pred_vertices = pred_vertices[0]
                pred_vertices, pred_3d_joints = pred_vertices + roots3d, pred_3d_joints + roots3d
                if coord_change_mat is not None:
                    pred_vertices = pred_vertices.matmul(coord_change_mat.to(pred_vertices.device))
                    pred_3d_joints = pred_3d_joints.matmul(coord_change_mat.to(pred_3d_joints.device))

        return att_output, hm_2d_joints, pred_3d_joints, pred_vertices, pred_camera, preds_obj, gt_mano_results


class HOModel_Trans(nn.Module):

    def __init__(self, honet, mano_model, mano_lambda_verts3d=None,
                 mano_lambda_joints3d=None,
                 mano_lambda_manopose=None,
                 mano_lambda_manoshape=None,
                 mano_lambda_regulshape=None,
                 mano_lambda_regulpose=None,
                 lambda_joints2d=None,
                 lambda_objects=None):

        super(HOModel_Trans, self).__init__()

        
        
        self.honet = honet
        self.mano_model = mano_model
        self.criterion_2d_keypoints = torch.nn.MSELoss(reduction='none')
        self.criterion_keypoints = torch.nn.MSELoss(reduction='none')
        self.criterion_vertices = torch.nn.L1Loss()
        # object loss
        self.object_loss = ObjectLoss(obj_reg_loss_weight=lambda_objects)

    def project(self, pred_verts, pred_3d_joints, pred_camera, gt_camera=None):
        pred_3d_joints_from_mesh = self.mano_model.get_3d_joints(pred_verts)
        #print(f"shape of gt_camera: {gt_camera.shape}")
        pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
        #pred_2d_joints_from_mesh = K_projection(pred_3d_joints_from_mesh.contiguous(), gt_camera)
        pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera.contiguous())
        #pred_2d_joints = K_projection(pred_3d_joints, gt_camera)
        #import pdb; pdb.set_trace()
        return pred_3d_joints_from_mesh, pred_2d_joints_from_mesh, pred_2d_joints

    def forward(self, imgs, bbox_hand, bbox_obj,mode = "train",
                joints_uv=None, joints_xyz=None, gt_camera=None, mano_params=None,  mask = None, roots3d=None,
                obj_p2d_gt=None, obj_mask=None, obj_lossmask=None, coord_change_mat=None):
        if mode != "evaluation":
            losses = {}
            total_loss = 0
            att_output, hm_2d_joints, pred_3d_joints, pred_vertices, pred_camera, preds_obj, gt_mano_results = self.honet(imgs, bbox_hand, bbox_obj, mano_params=mano_params, mask = mask, mode=mode)

            pred_verts_sub = pred_vertices[1]
            pred_verts = pred_vertices[0]

            pred_3d_joints_from_mesh, pred_2d_joints_from_mesh, pred_2d_joints = self.project(pred_verts, pred_3d_joints, pred_camera, gt_camera=gt_camera)
            # normalize the projected 2d joints to match gt scale
            #bbox_hand_expand = bbox_hand.view(-1, 1, 4)
            #print(bbox_hand_expand)
            #pred_2d_joints_from_mesh = (pred_2d_joints_from_mesh - bbox_hand_expand[:,:,:2]) / (bbox_hand_expand[:,:,2:] - bbox_hand_expand[:,:,:2])
            #pred_2d_joints = (pred_2d_joints - bbox_hand_expand[:,:,:2]) / (bbox_hand_expand[:,:,2:] - bbox_hand_expand[:,:,:2])

            if mano_params is not None:

                device = gt_mano_results["verts3d"].device
                
                verts_total_loss = 0.5 * Loss.vertices_loss(self.criterion_vertices.to(device), pred_verts, gt_mano_results["verts3d"]) + 0.5 * Loss.vertices_loss(self.criterion_vertices, pred_verts_sub, gt_mano_results["verts3d_down"])
                
                joint3d_mesh_loss = Loss.keypoint_3d_loss(self.criterion_keypoints.to(device), pred_3d_joints_from_mesh , gt_mano_results["joints3d"])
                joint3d_loss = Loss.keypoint_3d_loss(self.criterion_keypoints.to(device), pred_3d_joints, gt_mano_results["joints3d"])
                joint3d_total_loss = joint3d_mesh_loss + joint3d_loss
                
                total_loss += verts_total_loss
                total_loss += joint3d_total_loss
                losses["vertice"] = verts_total_loss
                losses["joint3d_mesh"] = joint3d_mesh_loss
                losses["joint3d"] = joint3d_loss

            if joints_uv is not None:
                device = joints_uv.device
                self.criterion_2d_keypoints.to(device)
                joint2d_loss = Loss.keypoint_2d_loss(self.criterion_2d_keypoints, pred_2d_joints, joints_uv)
                joint2d_mesh_loss = Loss.keypoint_2d_loss(self.criterion_2d_keypoints, pred_2d_joints_from_mesh, joints_uv)
                #joint2d_hm_loss = Loss.keypoint_2d_loss(self.criterion_2d_keypoints, hm_2d_joints, joints_uv)
                #joint2d_total_loss = joint2d_loss + joint2d_mesh_loss + 0.1*joint2d_hm_loss
                joint2d_total_loss = joint2d_loss + joint2d_mesh_loss
                total_loss += joint2d_total_loss
                losses["joint2d"] = joint2d_loss
                losses["joint2d_mesh"] = joint2d_mesh_loss
                #losses["joint2d_hm"] = joint2d_hm_loss

            if preds_obj is not None:
                obj_total_loss, obj_losses = self.object_loss.compute_loss(obj_p2d_gt, obj_mask, preds_obj, obj_lossmask=obj_lossmask)
                for key, val in obj_losses.items():
                    losses[key] = val
                total_loss += obj_total_loss
            
            if total_loss is not None:
                losses["total_loss"] = total_loss.detach().cpu()
            else:
                losses["total_loss"] = 0

            return total_loss, losses, pred_camera, pred_3d_joints, pred_verts, pred_2d_joints, preds_obj, gt_mano_results
        else:
            att_output, hm_2d_joints, pred_3d_joints, pred_vertices, pred_camera, preds_obj, gt_mano_results = self.honet.module.forward(imgs, bbox_hand, bbox_obj, roots3d=roots3d, coord_change_mat=coord_change_mat, mode=mode)
            #pred_verts = pred_vertices[0]
            pred_3d_joints_from_mesh, pred_2d_joints_from_mesh, pred_2d_joints = self.project(pred_vertices, pred_3d_joints, pred_camera)

            return pred_camera, pred_2d_joints, pred_3d_joints, pred_vertices, preds_obj


###############################################################################
class HONet(nn.Module):
    def __init__(self, roi_res=32, joint_nb=21, stacks=1, channels=256, blocks=1,
                 transformer_depth=1, transformer_head=8,
                 mano_layer=None, mano_neurons=[1024, 512], coord_change_mat=None,
                 reg_object=True, pretrained=True):

        super(HONet, self).__init__()

        self.out_res = roi_res

        # FPN-Res50 backbone
        self.base_net = FPN(pretrained=pretrained)

        # hand head
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb,
                                      stacks=stacks, channels=channels, blocks=blocks)
        #TODO: initialize hand_pre_encoder
        self.hand_pre_encoder = hand_feature_Extractor(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        #TODO: transformer encoder: self.hand_trans_encoder = hand_trans_Encoder()
        # hand encoder
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        # mano branch
        self.mano_branch = mano_regHead(mano_layer, feature_size=self.hand_encoder.num_feat_out,
                                        mano_neurons=mano_neurons, coord_change_mat=coord_change_mat)
        # object head
        self.reg_object = reg_object
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels//2, joint_nb=joint_nb)
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)

        # CR blocks
        self.transformer = Transformer(inp_res=roi_res, dim=channels,
                                       depth=transformer_depth, num_heads=transformer_head)
        #TODO: add hand template to features
        template_pose = torch.zeros((1,48)).cuda()
        template_betas = torch.zeros((1,10)).cuda()
        mano_layer = mano_layer.cuda()
        template_vertices, template_3d_joints = mano_layer(template_pose, template_betas)
        # here use the mano_layer revised for ho3d (may reach better result)
        self.template_vertices = template_vertices/1000.0
        self.template_3d_joints = template_3d_joints/1000.0
        # concatinate template joints and template vertices, and then duplicate to batch size
        


    def net_forward(self, imgs, bbox_hand, bbox_obj, mano_params=None, roots3d=None):
        batch = imgs.shape[0]
        inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
        inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
        bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1)
        msk_inter = ((inter_bottomRight-inter_topLeft > 0).sum(dim=1)) == 2
        # P2 from FPN Network
        P2 = self.base_net(imgs)[0]
        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)
        # get roi boxes
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        # 4 here is the downscale size in FPN network(P2)
        x = ops.roi_align(P2, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand
        #print(f"check hand roi shape: {x.shape}")
        
        #print(f"")
        # hand forward
        out_hm, encoding, preds_joints = self.hand_head(x)
        #print(f"check heatmap shape: {out_hm[-1].shape}; check encoding vector shape: {encoding[-1].shape}")
        #TODO:transformer branch - forward hand_pre_encoder for image_feat and grid_feat
        image_feat, grid_feat = self.hand_pre_encoder(out_hm, encoding)
        
        batch_size = image_feat.shape[0]
        ref_vertices = torch.cat([self.template_3d_joints, self.template_vertices],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)
        ref_joints = self.template_3d_joints.expand(batch_size, -1, -1)

        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        #print(f"check shape of image_feat:{image_feat.shape}")
        #TODO: concatinate image feat and template mesh to form the joint/vertex queries
        features = torch.cat([ref_vertices.to(image_feat.device), image_feat], dim=2)

        #TODO: concatenate template joints to grid_features to form 'joint_grid_feat'
        joint_grid_feat = torch.cat([ref_joints.to(grid_feat.device), grid_feat], dim=2)
        #print(f"check shape of grid_features:{joint_grid_feat.shape}")

        #TODO: prepare input tokens including joint/vertex queries and jotin_grid_features
        features = torch.cat([features, joint_grid_feat],dim=1)
        #print(f"check shape of features:{features.shape}")

        #mano branch
        mano_encoding = self.hand_encoder(out_hm, encoding)
        #print(f"check mano_encoding shape: {mano_encoding.shape}")

        pred_camera, pred_mano_results, gt_mano_results = self.mano_branch(mano_encoding, mano_params=mano_params, roots3d=roots3d)
        #print(f"check shape of pred_camera: \n {pred_camera.shape}")

        # obj forward
        if self.reg_object:
            roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)
            roi_boxes_inter = torch.cat((idx_tensor, bbox_inter), dim=1)

            y = ops.roi_align(P2, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # obj

            z = ops.roi_align(P2, roi_boxes_inter, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                              sampling_ratio=-1)  # intersection
            z = msk_inter[:, None, None, None] * z
            y = self.transformer(y, z)
            out_fm = self.obj_head(y)
            preds_obj = self.obj_reorgLayer(out_fm)
        else:
            preds_obj = None
        return pred_camera, preds_joints, pred_mano_results, gt_mano_results, preds_obj

    def forward(self, imgs, bbox_hand, bbox_obj, mano_params=None, roots3d=None):
        if self.training: # 'training' is the attribute of nn.Module which can be configured by model.train() and model.eval()
            pred_camera, preds_joints, pred_mano_results, gt_mano_results, preds_obj = self.net_forward(imgs, bbox_hand,    bbox_obj, mano_params=mano_params)
            return pred_camera, preds_joints, pred_mano_results, gt_mano_results, preds_obj
        else:
            pred_camera, preds_joints, pred_mano_results, _, preds_obj = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                             roots3d=roots3d)
            return pred_camera, preds_joints, pred_mano_results, preds_obj


class HOModel(nn.Module):

    def __init__(self, honet, mano_lambda_verts3d=None,
                 mano_lambda_joints3d=None,
                 mano_lambda_manopose=None,
                 mano_lambda_manoshape=None,
                 mano_lambda_regulshape=None,
                 mano_lambda_regulpose=None,
                 lambda_joints2d=None,
                 lambda_objects=None):

        super(HOModel, self).__init__()
        self.honet = honet
        # supervise when provide mano params
        self.mano_loss = ManoLoss(lambda_verts3d=mano_lambda_verts3d,
                                  lambda_joints3d=mano_lambda_joints3d,
                                  lambda_manopose=mano_lambda_manopose,
                                  lambda_manoshape=mano_lambda_manoshape)
        self.joint2d_loss = Joint2DLoss(lambda_joints2d=lambda_joints2d)
        # supervise when provide hand joints
        self.mano_joint_loss = ManoLoss(lambda_joints3d=mano_lambda_joints3d,
                                        lambda_regulshape=mano_lambda_regulshape,
                                        lambda_regulpose=mano_lambda_regulpose)
        # object loss
        self.object_loss = ObjectLoss(obj_reg_loss_weight=lambda_objects)

    def forward(self, imgs, bbox_hand, bbox_obj,
                joints_uv=None, joints_xyz=None, mano_params=None, roots3d=None,
                obj_p2d_gt=None, obj_mask=None, obj_lossmask=None):
        if self.training:
            losses = {}
            total_loss = 0
            pred_camera, preds_joints2d, pred_mano_results, gt_mano_results, preds_obj= self.honet(imgs, bbox_hand, bbox_obj, mano_params=mano_params)
            if mano_params is not None:
                mano_total_loss, mano_losses = self.mano_loss.compute_loss(pred_mano_results, gt_mano_results)
                total_loss += mano_total_loss
                for key, val in mano_losses.items():
                    losses[key] = val
            if joints_uv is not None:
                joint2d_loss, joint2d_losses = self.joint2d_loss.compute_loss(preds_joints2d, joints_uv)
                for key, val in joint2d_losses.items():
                    losses[key] = val
                total_loss += joint2d_loss
            if preds_obj is not None:
                obj_total_loss, obj_losses = self.object_loss.compute_loss(obj_p2d_gt, obj_mask, preds_obj, obj_lossmask=obj_lossmask)
                for key, val in obj_losses.items():
                    losses[key] = val
                total_loss += obj_total_loss
            if total_loss is not None:
                losses["total_loss"] = total_loss.detach().cpu()
            else:
                losses["total_loss"] = 0
            return total_loss, losses, pred_camera, preds_joints2d, pred_mano_results, preds_obj
        else:
            pred_camera, preds_joints2d, pred_mano_results, _, preds_obj = self.honet.module.net_forward(imgs, bbox_hand, bbox_obj, roots3d=roots3d)
            return pred_camera, preds_joints2d, pred_mano_results, preds_obj


