from __future__ import print_function, unicode_literals
import numpy as np
import json
import os
import time
import skimage.io as io
import pickle
import math
import sys
import matplotlib.pyplot as plt
import cv2

""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def show2DBoundingBox(imgInOrg, bb):
    """ Show bounding box on the image"""
    imgIn = np.copy(imgInOrg)
    imgIn = cv2.rectangle(imgIn, (int(bb[0]), int(bb[1])),
                          (int(bb[2]), int(bb[3])), (0, 0, 255), thickness=3)
    return imgIn

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
                        [70, 17, 145]] # pinky

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
            if joint_num in [0, 4, 8, 12, 16]: 
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
    jointColsGt = (255,255,0)
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

    if filename is not None:
        cv2.imwrite(filename, img)

    return img

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def plot3dVisualize(ax, m, flip_x=False, c="b", alpha=0.1, camPose=np.eye(4, dtype=np.float32), isOpenGLCoords=False):
    '''
    Create 3D visualization
    :param ax: matplotlib axis
    :param m: mesh
    :param flip_x: flix x axis?
    :param c: mesh color
    :param alpha: transperency
    :param camPose: camera pose
    :param isOpenGLCoords: is mesh in openGL coordinate system?
    :return:
    '''
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if hasattr(m, 'r'):
        verts = np.copy(m.r)*1000
    elif hasattr(m, 'v'):
        verts = np.copy(m.v) * 1000
    else:
        raise Exception('Unknown Mesh format')
    vertsHomo = np.concatenate([verts, np.ones((verts.shape[0],1), dtype=np.float32)], axis=1)
    verts = vertsHomo.dot(camPose.T)[:,:3]

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        verts = verts.dot(coordChangeMat.T)

    faces = np.copy(m.f)
    ax.view_init(elev=90, azim=-90)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == "b":
        face_color = (141 / 255, 184 / 255, 226 / 255)
        face_color = np.tile(np.array([[0., 0., 1., 1.]]), [verts.shape[0], 1])
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "r":
        face_color = (226 / 255, 141 / 255, 141 / 255)
        face_color = np.tile(np.array([[1., 0., 0., 1.]]), [verts.shape[0], 1])
        edge_color = (112 / 255, 0 / 255, 0 / 255)
    elif c == "viridis":
        face_color = plt.cm.viridis(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "plasma":
        face_color = plt.cm.plasma(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        # edge_color = (0 / 255, 0 / 255, 112 / 255)
    else:
        face_color = c
        edge_color = c

    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts, flip_x=flip_x)
    # plt.tight_layout()

class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs
        


class Open3DWin():
    def __init__(self):
        import open3d
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name='Open3D', width=640, height=480, left=0, top=0,
                          visible=True)  # use visible=True to visualize the point cloud
        # vis.get_render_option().light_on = False
        self.vis.get_render_option().mesh_show_back_face = True
        #self.vis.get_render_option().load_from_json("render_options.json")
        
    def capture_view(self, mesh, view_mat_path=None,intrinsics=None):
        
        if not isinstance(view_mat_path, np.ndarray) and view_mat_path is not None:
            assert os.path.exists(view_mat_path)
            view_mat = np.loadtxt(view_mat_path)
        else:
            view_mat = view_mat_path
    
        camera_param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        cx = camera_param.intrinsic.intrinsic_matrix[0, 2]
        cy = camera_param.intrinsic.intrinsic_matrix[1, 2]
    
        if intrinsics is not None:
            camera_param.intrinsic.set_intrinsics(camera_param.intrinsic.width, camera_param.intrinsic.height,
                                                  intrinsics[0, 0], intrinsics[1, 1], cx, cy)
    
        if view_mat is not None:
            camera_param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            camera_param.extrinsic = view_mat
    
        ctr = self.vis.get_view_control()
        # ctr.set_constant_z_far(20.)
        # ctr.set_constant_z_near(-2)
        for m in mesh:
            self.vis.add_geometry(m)
    
        ctr.convert_from_pinhole_camera_parameters(camera_param)
    
    
    
        # vis.run()
    
        render = self.vis.capture_screen_float_buffer(do_render=True)
    
        render = (np.asarray(render)*255).astype(np.uint8)

        for m in mesh:
            self.vis.remove_geometry(m)
    
        return render

def open3dVisualize(mList, colorList):
    import open3d
    o3dMeshList = []
    for i, m in enumerate(mList):
        mesh = open3d.geometry.TriangleMesh()
        numVert = 0
        if hasattr(m, 'r'):
            mesh.vertices = open3d.utility.Vector3dVector(np.copy(m.r))
            numVert = m.r.shape[0]
        elif hasattr(m, 'v'):
            mesh.vertices = open3d.utility.Vector3dVector(np.copy(m.v))
            numVert = m.v.shape[0]
        else:
            raise Exception('Unknown Mesh format')
        mesh.triangles = open3d.utility.Vector3iVector(np.copy(m.f))
        if colorList[i] == 'r':
            mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.6, 0.2, 0.2]]), [numVert, 1]))
        elif colorList[i] == 'g':
            mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))
        elif isinstance(colorList[i],np.ndarray):
            assert colorList[i].shape == np.array(mesh.vertices).shape
            mesh.vertex_colors = open3d.utility.Vector3dVector(colorList[i])
        else:
            raise Exception('Unknown mesh color')

        o3dMeshList.append(mesh)
    #open3d.visualization.draw_geometries(o3dMeshList)
    return o3dMeshList

def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'vn': [], 'f': [], 'vt': [], 'ft': [], 'fn': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])
            if len(spl[0]) > 1 and spl[1] and 'ft' in d:
                d['ft'].append([np.array([int(l[1])-1 for l in spl[:3]])])
            if len(spl[0]) > 2 and spl[2] and 'fn' in d:
                d['fn'].append([np.array([int(l[2])-1 for l in spl[:3]])])

            # TOO: redirect to actual vert normals?
            #if len(line[0]) > 2 and line[0][2]:
            #    d['fn'].append([np.concatenate([l[2] for l in spl[:3]])])
        elif key == 'vn':
            d['vn'].append([np.array([float(v) for v in values])])
        elif key == 'vt':
            d['vt'].append([np.array([float(v) for v in values])])


    for k, v in d.items():
        if k in ['v','vn','f','vt','ft', 'fn']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result


def db_size(set_name, version='v2'):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        if version == 'v2':
            return 66034  # number of unique samples (they exists in multiple 'versions')
        elif version == 'v3':
            return 78297
        else:
            raise NotImplementedError
    elif set_name == 'evaluation':
        if version == 'v2':
            return 11524
        elif version == 'v3':
            return 20137
        else:
            raise NotImplementedError
    else:
        assert 0, 'Invalid choice.'

def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts

def read_RGB_img(base_dir, seq_name, file_id, split):
    """Read the RGB image in dataset"""
    if os.path.exists(os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.png')):
        img_filename = os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.png')
    else:
        img_filename = os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.jpg')

    _assert_exist(img_filename)

    img = cv2.imread(img_filename)

    return img

def read_seg(base_dir, seq_name, file_id, split):
    """Read the RGB image in dataset"""
    img_filename = os.path.join(base_dir, split, seq_name, 'segr', file_id + '.png')

    _assert_exist(img_filename)

    img = cv2.imread(img_filename)

    return img


def read_depth_img(base_dir, seq_name, file_id, split):
    """Read the depth image in dataset and decode it"""
    depth_filename = os.path.join(base_dir, split, seq_name, 'depth', file_id + '.png')

    _assert_exist(depth_filename)

    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt * depth_scale

    return dpt

def read_annotation(base_dir, seq_name, file_id, split):
    meta_filename = os.path.join(base_dir, split, seq_name, 'meta', file_id + '.pkl')

    _assert_exist(meta_filename)

    pkl_data = load_pickle_data(meta_filename)

    return pkl_data


