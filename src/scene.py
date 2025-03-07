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
