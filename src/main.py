import os, sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import rootutils
from logging import Logger
from utils_ema.log import get_logger_default
from camera_calibrator import CameraCalibrator

# load conf with hydra and run
@hydra.main(version_base=None)
def main(cfg: DictConfig):

    # resolve paths
    os.environ["ROOT"] =str(os.getcwd()) 
    OmegaConf.resolve(cfg)

    # init logger
    logger = get_logger_default(out_path=cfg.paths.log_file)

    # run the program
    logger.info("Program started.")
    run(cfg, logger)
    logger.info("Program ended.")


# run the program
def run(cfg: DictConfig, logger: Logger):
    c = CameraCalibrator(logger=logger, cfg = cfg)
    c.collect_images()
    pass


if __name__ == "__main__":
    main()
