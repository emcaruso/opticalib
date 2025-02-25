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
from objects.charuco import CharucoObjects


class CameraCalibrator():

    def __init__(self, cfg : DictConfig, logger : Logger):
        self.cfg = cfg
        self.logger = logger
        self.load_objects()

    def load_objects(self):
        assert( "type" in self.cfg.objects )
        if self.cfg.objects.type == "charuco":
            self.objects = CharucoObjects(self.cfg.objects)
        # add more objects here in the future
        
    def collect_images(self):
        c = Collector( cfg = self.cfg.collectors, logger = self.logger)
        c.postprocessing.add_function(self.objects.draw_features)
        if self.cfg.calibration.collect.mode.val == "manual":
            c.capture_manual(in_ram = self.cfg.calibration.collect.in_ram)
            c.save()
        elif self.cfg.calibration.collect.mode.val == "automatic":
            c.capture_till_q(in_ram = self.cfg.calibration.collect.in_ram)
            c.save()

    def calibrate(self):
        c = Charuco(self.cfg.calibration)
        c.calibrate()
