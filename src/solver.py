import numpy as np
from tqdm import tqdm
import torch
from typing import List, Dict
from logging import Logger
from omegaconf import DictConfig
from hydra.utils import instantiate
from objects.object import Object, Features, ObjectDetector
from sensorflow.src.collector import Collector
from utils_ema.camera_cv import Camera_cv
import torch


class Dataset:

    def __init__(self, obj: Object, features_gt: List[List[Features]]):
        self.obj = obj
        self.points_gt: List[List[List[torch.Tensor]]] = [
            [f.points() for f in f_list] for f_list in features_gt
        ]
        self.ids_gt: List[List[List[torch.Tensor]]] = [
            [f.ids() for f in f_list] for f_list in features_gt
        ]

    def __getitem__(self, idx):
        time_id = idx
        features_list = self.features_gt[time_id]
        for cam_id, f in enumerate(features_list):
            for board_id, ids_gt in enumerate(self.ids_gt):
                ids_gt = f.ids()
                ids_hat = self.obj.ids[cam_id]

    def __len__(self):
        pass


class Solver:
    def __init__(self, cfg: DictConfig, obj: Object, logger: Logger):
        self.logger.info("Initializing calibration...")
        self.logger: Logger = logger
        self.cfg: DictConfig = cfg
        self.features_gt: List[List[Features]] = self.__collect_features_gt()
        self.time_instants: int = len(self.features_gt)
        self.n_cameras: int = len(self.features_gt[0])
        self.obj = obj
        self.objects = self.__get_objects()
        self.cameras: List[Camera_cv] = self.__get_cameras()
        self.obj_parameters: List[torch.Tensor] = self.__collect_object_parameters()
        self.cam_parameters: List[torch.Tensor] = self.__collect_cameras_parameters()
        self.optimizer = instantiate(
            self.cfg.optimizer, params=self.obj_parameters + self.cam_parameters
        )
        self.scheduler = instantiate(self.cfg.scheduler, optimizer=self.optimizer)

    def run(self):
        self.logger.info("Calibrating...")
        progress_bar = tqdm(range(self.cfg.calibration.iterations), desc="Iteration: ")
        for _ in progress_bar:

            # for each time instant
            loss_total = 0
            for time_id in range(self.time_instants):
                f_hat = self.__project_features(time_id)
                loss = self.__loss(f_hat, self.features_gt[time_id])
                loss_total += loss.item()
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            progress_bar.set_postfix({"loss": loss_total})

    def __get_objects(self) -> List[Object]:
        for time_id in range(self.time_instants):
            yield self.obj.clone()
        pass

    def __loss(self, f_hat, f_gt):
        pass

    def __collect_features_gt(self) -> List[List[Features]]:
        features = []
        coll_data = Collector.load(self.cfg.collector.paths.save_dir, in_ram=False)
        for images in coll_data.raw_images():
            features.append(self.obj.detector.detect_features(images))
        return features

    def __project_features(self, time_id: int):
        pass

    def __get_cameras(self) -> List[Camera_cv]:
        cameras = []
        for _ in range(self.n_cameras):
            cameras.append(Camera_cv())
        return cameras

    def __collect_object_parameters(self, obj):
        pass

    def __collect_cameras_parameters(self, obj):
        pass

    def save(self):
        pass

    def show_results(self):
        pass
