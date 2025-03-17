import itertools
import torch
from objects.object import Object, ObjectDetector, Features
from omegaconf import DictConfig
from logging import Logger
from typing import List, Tuple
from utils_ema.camera_cv import Camera_cv, Intrinsics
from utils_ema.geometry_pose import Pose
from utils_ema.image import Image
from utils_ema.geometry_euler import eul


class Scene():

    def __init__(self, cfg: DictConfig, logger: Logger, features_gt: List[List[Features]], cameras: List[Camera_cv], objects: List[Object]):
        self.cfg = cfg
        self.logger = logger
        self.features_gt = features_gt
        self.cameras = cameras
        self.objects = objects
        self.world_pose = Pose(euler = eul(torch.zeros([3], dtype=torch.float32)), position = torch.zeros([3], dtype=torch.float32))
        self.n_cameras = len(self.features_gt[0])
        self.time_instants = len(self.features_gt)
        self.n_features_min = int(self.cfg.calibration.percentage_points_min*self.objects[0].params.n_corners)


    def get_xy(self, time_id: int, concatenate: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:

        points_gt_list = []
        points_hat_list = []
        obj = self.objects[time_id]
        ids_list_hat = obj.ids()
        points3D_list_hat = obj.points()
        # points3D_list_hat = [p @ self.world_pose.rotation().transpose(-2, -1) for p in obj.points() ]
        features_list = self.features_gt[time_id]
        # for cam_id in range(self.n_cameras)[1:]:
        for cam_id in range(self.n_cameras):
            points_gt_l = []
            points_hat_l = []
            f = features_list[cam_id]
            ids_list_gt = f.ids
            points_list_gt = f.points
            cam = self.cameras[cam_id] 
            cam.intr.update_intrinsics()
            for board_id in range(len(ids_list_gt)):
                ids_gt = ids_list_gt[board_id]
                if len(ids_gt) < self.n_features_min:
                    continue
                ids_hat = ids_list_hat[board_id]
                points_gt = points_list_gt[board_id]
                points3D_hat = points3D_list_hat[board_id]
                idxs = torch.where(torch.isin(ids_hat, ids_gt.squeeze(-1)))[0]
                points3D_hat = points3D_hat[idxs]
                points2D_hat_undistort = cam.project_points(points3D_hat, longtens=False, und=False, transform_cam_pose=self.world_pose)
                points_hat = cam.distort(points2D_hat_undistort)
                # points_hat = points2D_hat_undistort 
                points_gt_l.append(points_gt)
                points_hat_l.append(points_hat)
            if points_gt_l != []:
                points_gt_list.append(torch.cat(points_gt_l, dim=0).squeeze(1))
                points_hat_list.append(torch.cat(points_hat_l, dim=0))
            else:
                points_gt_list.append(torch.zeros([0,2]))
                points_hat_list.append(torch.zeros([0,2]))

        if len(points_hat_list) == 0:
            return torch.tensor(points_hat_list), torch.tensor(points_hat_list)

        if concatenate:
            return torch.cat(points_hat_list, dim=0), torch.cat(points_gt_list, dim=0)
        else:
            return points_hat_list, points_gt_list

    def get_3D_2D_points_foreach_cam(self, filter_boards_by_n_features_min: bool = False, numpy: bool = False):
        points_3D = []
        points_2D = []
        infos = []
        ids3D = self.objects[0].ids()

        n_corners = self.objects[0].params.n_corners

        # for each cam
        for cam_id in range(len(self.features_gt[0])):

            ratio = self.cameras[cam_id].intr.pixel_unit_ratio()

            # init relative poses
            points_2D_list = []
            points_3D_list = []
            infos_list = []

            for board_id in range(self.objects[0].params.n_boards):
                # ids_3D_list = ids3D[board_id]

                for time_id in range(len(self.features_gt)):
                    
                    features = self.features_gt[time_id][cam_id]
                    ids_2D = features.ids[board_id]

                    if filter_boards_by_n_features_min and len(ids_2D) < self.n_features_min:
                        continue

                    p2D = features.points[board_id]
                    p3D = self.objects[time_id].params.points_list[board_id][ids_2D-board_id*n_corners].squeeze(1)
                    points_2D_list.append(p2D.squeeze(1).cpu().numpy())
                    points_3D_list.append((p3D * ratio).cpu().numpy())
                    infos_list.append({"cam_id": cam_id, "board_id": board_id, "time_id": time_id})

            points_3D.append(points_3D_list)
            points_2D.append(points_2D_list)
            infos.append(infos_list)

        if numpy:
            points_3D = [p3D for p3D in points_3D]
            points_2D = [p for p in points_2D]

        return points_3D, points_2D, infos

