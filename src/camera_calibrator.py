import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from pathlib import Path
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
            self.obj = CharucoObject.init_base(self.cfg.objects, device=self.cfg.calibration.device)
            self.detector = CharucoDetector(self.obj.params)
        # add more objects here in the future

    def collect_images(self):

        # ask user to confirm
        user_input = input("Are you sure you want to start collecting images? (y/n): ")
        if user_input != "y":
            self.logger.info("Exiting program.")
            sys.exit()

        c = Collector(cfg=self.cfg.collector, logger=self.logger)
        c.postprocessing.add_function(self.detector.draw_features)
        self.__set_lights(c)
        if self.cfg.collector.mode.val == "manual":
            c.capture_manual(in_ram=False)
        elif self.cfg.collector.mode.val == "automatic":
            trigger = self.detector.images_has_at_least_one_feature
            c.capture_till_q(in_ram=False, trigger=trigger)
        c.save(save_raw = True, save_postprocessed = True)
        self.__lights_off(c)

    def __set_lights(self, c):
        if c.light_controller is not None:
            c.light_controller.leds_off()
            for channel in self.cfg.collector.lights.channels:
                c.light_controller.led_on(channel, amp=self.cfg.collector.lights.intensity)

    def __lights_off(self, c):
        if c.light_controller is not None:
            c.light_controller.leds_off()

    def calibrate(self):

        s = Solver(
            cfg=self.cfg,
            obj=self.obj,
            detector=self.detector,
            logger=self.logger,
        )
        s.calibrate()
        s.save_colormaps()
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
