import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.linalg import orthogonal_procrustes


def verts2pcd(verts, color=None):
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(verts)
    if color is not None:
        if color == 'r':
            pcd.paint_uniform_color([1, 0.0, 0])
        if color == 'g':
            pcd.paint_uniform_color([0, 1.0, 0])
        if color == 'b':
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)
    d1 = o3d.compute_point_cloud_to_point_cloud_distance(gt, pr) # closest dist for each gt point
    d2 = o3d.compute_point_cloud_to_point_cloud_distance(pr, gt) # closest dist for each pred point
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))  # how many of our predicted points lie close to a gt point?
        precision = float(sum(d < th for d in d1)) / float(len(d1))  # how many of gt points are matched?

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall

def align_sc_tr(mtx1, mtx2):
    """ Align the 3D joint location with the ground truth by scaling and translation """

    predCurr = mtx2.copy()
    # normalize the predictions
    s = np.sqrt(np.sum(np.square(predCurr[4] - predCurr[0])))
    if s>0:
        predCurr = predCurr / s

    # get the scale of the ground truth
    sGT = np.sqrt(np.sum(np.square(mtx1[4] - mtx1[0])))

    # make predictions scale same as ground truth scale
    predCurr = predCurr * sGT

    # make preditions translation of the wrist joint same as ground truth
    predCurrRel = predCurr - predCurr[0:1, :]
    preds_sc_tr_al = predCurrRel + mtx1[0:1, :]

    return preds_sc_tr_al




def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def align_by_trafo(mtx, trafo):
    t2 = mtx.mean(0)
    mtx_t = mtx - t2
    R, s, s1, t1 = trafo
    return np.dot(mtx_t, R.T) * s * s1 + t1 + t2


class curve:
    def __init__(self, x_data, y_data, x_label, y_label, text):
        self.x_data = x_data
        self.y_data = y_data
        self.x_label = x_label
        self.y_label = y_label
        self.text = text

class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred, skip_check=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if not skip_check:
            keypoint_gt = np.squeeze(keypoint_gt)
            keypoint_pred = np.squeeze(keypoint_pred)
            keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

            assert len(keypoint_gt.shape) == 2
            assert len(keypoint_pred.shape) == 2
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds
    
def calculate_results(eval_xyz, eval_xyz_procrustes_aligned, eval_mesh_err, eval_mesh_err_aligned, txt_file, xyz_dict, mesh_dict, epoch):
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    #print('Evaluation 3D KP results:')
    #print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

    xyz_procrustes_al_mean3d, _, xyz_procrustes_al_auc3d, pck_xyz_procrustes_al, thresh_xyz_procrustes_al = eval_xyz_procrustes_aligned.get_measures(0.0, 0.05, 100)

    mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(0.0, 0.05, 100)

    mesh_al_mean3d, _, mesh_al_auc3d, pck_mesh_al, thresh_mesh_al = eval_mesh_err_aligned.get_measures(0.0, 0.05, 100)

    # -- print out in txt
    print(f"\n Start printing out the metrics results in {txt_file} ...")
    f = open(txt_file, "a")
    f.write(f"================= Epoch: {epoch}===========\n")
    f.write(f"Evaluation 3D KP results:\n auc={xyz_auc3d:.3f}, mean_vert3d_avg={xyz_mean3d * 100.0:.2f} cm\n")
    f.write(f"Evaluation 3D KP PROCRUSTES ALIGNED results:\n auc={xyz_procrustes_al_auc3d:.3f}, mean_vert3d_avg={xyz_procrustes_al_mean3d * 100.0:.2f} cm\n")
    f.write(f"Evaluation 3D MESH results:\n auc={mesh_auc3d:.3f}, mean_vert3d_avg={mesh_mean3d * 100.0:.2f} cm\n")
    f.write(f"Evaluation 3D MESH PROCRUSTES ALIGNED results:\n auc={mesh_al_auc3d:.3f}, mean_vert3d_avg={mesh_al_mean3d * 100.0:.2f} cm\n")
    f.close()
    print(f"\n txt writing done!")


    # # append the lists in the
    xyz_dict["auc"].append(xyz_auc3d)
    xyz_dict["mean"].append(xyz_mean3d * 100.0) # all mean in cm
    xyz_dict["al_auc"].append(xyz_procrustes_al_auc3d)
    xyz_dict["al_mean"].append(xyz_procrustes_al_mean3d * 100.0)

    mesh_dict["auc"].append(mesh_auc3d)
    mesh_dict["mean"].append(mesh_mean3d * 100.0)
    mesh_dict["al_auc"].append(mesh_al_auc3d)
    mesh_dict["al_mean"].append(mesh_al_mean3d * 100.0)
        
    # measure elapsed time

    return xyz_dict, mesh_dict

    
