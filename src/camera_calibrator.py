import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from pathlib import Path
from logging import Logger
from omegaconf import DictConfig
from sensorflow.src.collector import Collector
from solver import Solver
from objects.charuco import CharucoObject, CharucoDetector
from utils_ema.image import Image


class CameraCalibrator:

    def __init__(self, cfg: DictConfig, logger: Logger):
        self.cfg = cfg
        self.logger = logger
        self.load_objects()

    def load_objects(self):
        assert "type" in self.cfg.objects
        if self.cfg.objects.type == "charuco":
            self.obj = CharucoObject.init_base(self.cfg.objects, device=self.cfg.calibration.device)
            self.detector = CharucoDetector(self.obj.params)
        # add more objects here in the future

    def collect_images(self):
        c = Collector(cfg=self.cfg.collector, logger=self.logger)
        c.postprocessing.add_function(self.detector.draw_features)
        self.__set_lights(c)
        if self.cfg.collect.mode.val == "manual":
            c.capture_manual(in_ram=self.cfg.collect.in_ram)
        elif self.cfg.collect.mode.val == "automatic":
            trigger = self.detector.images_has_at_least_one_feature
            c.capture_till_q(in_ram=self.cfg.collect.in_ram, trigger=trigger)
        c.save(save_raw = True, save_postprocessed = True)
        c.light_controller.leds_off()

    def __set_lights(self, c):
        c.light_controller.leds_off()
        for channel in self.cfg.collect.lights.channels:
            c.light_controller.led_on(channel, amp=self.cfg.collect.lights.intensity)

    def calibrate(self):

        s = Solver(
            cfg=self.cfg,
            obj=self.obj,
            detector=self.detector,
            logger=self.logger,
        )
        s.calibrate()
        s.plot_colormaps()
        s.save()

    def generate_charuco_images(self, show=False):
        images = self.obj.generate_charuco_images()
        dir = Path(self.cfg.paths.charuco_images_dir)
        if not dir.exists():
            os.makedirs(self.cfg.paths.charuco_images_dir)

        # if show:
        #     Image.show_multiple_images(images, wk = 0)
        #
        self.logger.info(f"Saving charuco images to {dir}")
        for id, img in enumerate(images):
            img.save(dir / f"{id:03d}.png")
