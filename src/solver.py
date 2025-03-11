import torch
import subprocess
import numpy as np

# from blender_saver import blender_save
from tqdm import tqdm
from typing import List, Tuple
from logging import Logger
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate
from objects.object import Object, Features, ObjectDetector
from blender_saver import blender_save
from optimizer import Optimizer
from initializer import Initializer
from utils_ema.plot import plotter


class Solver:
    def __init__(
        self, cfg: DictConfig, obj: Object, detector: ObjectDetector, logger: Logger
    ) -> None:
        self.logger: Logger = logger
        self.cfg: DictConfig = cfg
        self.initializer = Initializer(cfg, logger, detector, obj)

    def run(self) -> None:

        # initialize scene
        self.logger.info("Initializing scene")
        self.scene = self.initializer.initialize()

        # calibrate scene
        self.logger.info("Calibrating...")
        # optimizer = Optimizer(cfg=self.cfg, scene=self.scene, logger=self.logger,
        #                     intr_K=True, intr_D=True, extr=False, obj_rel=True, obj_pose=False, n_features_min=scene.n_features_min)
        # optimizer.run()

    def save(self) -> None:
        # Save calibration results on blender
        blender_save(self.cfg.paths.calib_results_dir, self.scene, self.logger)
 
    def load(self) -> bool:
        return True
