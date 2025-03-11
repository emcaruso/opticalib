from utils_ema.camera_cv import Camera_cv
import torch
from objects.object import Object, ObjectDetector, Features
from omegaconf import DictConfig
from logging import Logger
from typing import List
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
        self.n_cameras = len(self.features_gt[0])
        self.time_instants = len(self.features_gt)
        self.n_features_min = int(self.cfg.calibration.percentage_points_min*self.objects[0].params.n_corners)

    def get_3D_2D_points_foreach_cam(self, filter_by_n_features_min: bool = False, numpy: bool = False):
        points_3D = []
        points = []
        infos = []

        n_corners = self.objects[0].params.n_corners
        for cam_id in range(len(self.features_gt[0])):
            self.logger.info(f"Calibrating intrinsics of camera {cam_id} with Opencv")

            ratio = self.cameras[cam_id].intr.pixel_unit_ratio()

            # init relative poses
            points_list = []
            points_3D_list = []
            infos_list = []

            for board_id in range(self.objects[0].params.n_boards):
                for time_id in range(len(self.features_gt)):
                    
                    features = self.features_gt[time_id][cam_id]
                    ids_2D = features.ids[board_id]

                    if filter_by_n_features_min and len(ids_2D) < self.n_features_min:
                        continue

                    p2D = features.points[board_id]
                    p3D = self.objects[time_id].params.points_list[board_id][ids_2D-board_id*n_corners].squeeze(1)
                    points_list.append(p2D.squeeze(1).cpu().numpy())
                    points_3D_list.append((p3D * ratio).cpu().numpy())
                    infos_list.append({"cam_id": cam_id, "board_id": board_id, "time_id": time_id})

            points_3D.append(points_3D_list)
            points.append(points_list)
            infos.append(infos_list)

        if numpy:
            points_3D = [p3D for p3D in points_3D]
            points = [p for p in points]

        return points_3D, points, infos
