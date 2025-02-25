import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from logging import Logger
from omegaconf import DictConfig
from sensorflow.src.collector import Collector
from solver import Solver
from objects.charuco import CharucoObject, CharucoDetector


class CameraCalibrator:

    def __init__(self, cfg: DictConfig, logger: Logger):
        self.cfg = cfg
        self.logger = logger
        self.load_objects()

    def load_objects(self):
        assert "type" in self.cfg.objects
        if self.cfg.objects.type == "charuco":
            self.obj = CharucoObject.init_base(self.cfg.objects)
            self.detector = CharucoDetector(self.obj.params)
        # add more objects here in the future

    def collect_images(self):
        c = Collector(cfg=self.cfg.collector, logger=self.logger)
        c.postprocessing.add_function(self.detector.draw_features)
        if self.cfg.collect.mode.val == "manual":
            c.capture_manual(in_ram=self.cfg.collect.in_ram)
            c.save()
        elif self.cfg.collect.mode.val == "automatic":
            c.capture_till_q(in_ram=self.cfg.collect.in_ram)
            c.save()

    def calibrate(self):

        c = Solver(
            cfg=self.cfg,
            obj=self.obj,
            detector=self.detector,
            logger=self.logger,
        )
        # c.run()
        c.save()
