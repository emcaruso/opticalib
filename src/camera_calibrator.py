import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from logging import Logger
from typing import Dict
from pathlib import Path
from omegaconf import DictConfig
from utils_ema.config_utils import load_yaml
from utils_ema.charuco import Charuco
from utils_ema.log import get_logger_default
from sensorflow.src.collector import Collector


class CameraCalibrator():

    def __init__(self, cfg : DictConfig, logger : Logger):
        self.cfg = cfg
        self.logger = logger

    def collect_images(self):
        c = Collector( cfg = self.cfg.collectors, logger = self.logger)
        imgs_raw, imgs_postpr = c.capture_manual()
        c.save(imgs_raw)
