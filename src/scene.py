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

    def __init__(self, cfg: DictConfig, logger: Logger, features_gt: List[List[Features]], cameras: Camera_cv, objects: List[Object]):
        self.cfg = cfg
        self.logger = logger
        self.features_gt = features_gt
        self.cameras = cameras
        self.objects = objects
        self.world_pose = Pose(euler = eul(torch.zeros([3], dtype=torch.float32)), position = torch.zeros([3], dtype=torch.float32))
        self.n_cameras = len(self.features_gt[0])
        self.time_instants = len(self.features_gt)
        self.n_features_min = int(self.cfg.calibration.percentage_points_min*self.objects.params.n_corners)


    def get_xy(self, pixel_unit=False) -> Tuple[torch.Tensor, torch.Tensor]:
        p3D, p2D = self.get_3D_2D_points()
        p2D_hat_und = self.cameras.project_points(p3D, longtens=False, und=False)
        p2D_hat = self.cameras.distort(p2D_hat_und)
        mask = (p2D==float('inf')).any(dim=-1)
        if pixel_unit:
            ratio = self.cameras.intr.pixel_unit_ratio()[...,None]
            return p2D * ratio, p2D_hat * ratio, mask
        return p2D, p2D_hat, mask

    def get_3D_2D_points(self):
        ratio = self.cameras.intr.pixel_unit_ratio()[...,None]
        p3D = self.objects.points()
        p2D = self.features_gt * (1/ratio)
        return p3D, p2D
