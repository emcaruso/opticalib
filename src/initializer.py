from omegaconf import DictConfig
from logging import Logger
from scene import Scene
from objects.object import Object, Features, ObjectDetector
from typing import List
from sensorflow.src.collector import CollectorLoader
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from optimizer import Optimizer
from utils_ema.camera_cv import Camera_cv, Intrinsics
from utils_ema.geometry_pose import Pose
from utils_ema.geometry_euler import eul


class Initializer():
    def __init__(self, cfg: DictConfig, logger: Logger, detector: ObjectDetector,  obj: Object):
        self.cfg = cfg
        self.logger = logger
        self.detector = detector
        self.obj = obj

    def initialize(self):

        # get scene
        scene = self.__get_scene()

        # intrinsics initialization
        self.__intrinsics_init(scene)
        
        # pose initialization

        return scene

    def __get_scene(self) -> Scene:
        self.features_gt = self.__collect_features_gt()
        self.cameras = self.__get_cameras()
        self.objects = self.__get_objects()
        scene = Scene(self.cfg, self.logger, self.features_gt, self.cameras, self.objects)
        return scene

    def __intrinsics_init(self, scene: Scene):


        for cam_id in range(len(self.features_gt[0])):

            self.logger.info(f"Calibrating intrinsics of camera {cam_id}")

            # reset poses
            scene.objects = self.__get_objects(same_relative_poses=True)

            # for obj in scene.objects:
            #     obj.pose = Pose(position=torch.tensor([0,0,-1], dtype=torch.float32), euler=eul(torch.zeros(3, dtype=torch.float32)))

            # align boards with cameras
            optim = Optimizer(cfg=self.cfg, scene=scene, logger=self.logger, cam_id=cam_id,
                            intr_K=False, intr_D=False, extr=False, obj_rel=True, obj_pose=True)

            optim.run()

            import ipdb; ipdb.set_trace()

            # align boards + intrinsics
            optim = Optimizer(cfg=self.cfg, scene=scene, logger=self.logger, cam_id=cam_id,
                            intr_K=True, intr_D=True, extr=False, obj_rel=True, obj_pose=True)
            optim.run()
            

    def __get_cameras(self) -> List[Camera_cv]:
        cameras = []
        n_cameras = len(self.features_gt[0])
        for i in range(n_cameras):

            # get resolution
            resolution = CollectorLoader.resolutions[i]
            info = CollectorLoader.load_info(self.cfg.paths.collection_dir)
            sensor_size = torch.FloatTensor(info[f"cam_{i:03d}"].SensorSize)*1e-3

            intr = Intrinsics(D=torch.zeros(5), resolution=resolution, sensor_size=sensor_size)
            pose = Pose(position=torch.tensor([0,0,-1], dtype=torch.float32), euler=eul(torch.zeros(3, dtype=torch.float32)))
            cameras.append(Camera_cv(device=self.cfg.calibration.device, intrinsics=intr, pose=pose))
        return cameras

    def __get_objects(self, same_relative_poses = True) -> List[Object]:
        time_instants = len(self.features_gt)
        objects = [
            self.obj.clone(same_pose=False, same_relative_poses=same_relative_poses)
            for _ in range(time_instants)
        ]
        return objects

    def __collect_features_gt(self) -> List[List[Features]]:
        self.coll_data = CollectorLoader.load_images(self.cfg.paths.collection_dir, in_ram=self.cfg.collect.in_ram, raw=True)
        next(self.coll_data) # initialize static members
        if not Path(self.cfg.paths.features_file).exists():
            features = []
            self.logger.info("Collecting features...")
            progress_bar = tqdm(self.coll_data, desc="Collecting features", total = CollectorLoader.n_images)
            for images in progress_bar:
                features.append(self.detector.detect_features(images, device=self.cfg.calibration.device))
            # save custom object with pickle
            np.save(self.cfg.paths.features_file, features, allow_pickle=True)

        else:
            features = np.load(self.cfg.paths.features_file, allow_pickle=True)
            for feat in features:
                for f in feat:
                    f.to(self.cfg.calibration.device)
        return features
