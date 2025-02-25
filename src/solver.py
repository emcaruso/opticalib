import torch
import subprocess

# from blender_saver import blender_save
from tqdm import tqdm
from typing import List, Tuple
from logging import Logger
from omegaconf import DictConfig
from hydra.utils import instantiate
from objects.object import Object, Features, ObjectDetector
from sensorflow.src.collector import Collector
from utils_ema.camera_cv import Camera_cv


class Solver:
    def __init__(
        self, cfg: DictConfig, obj: Object, detector: ObjectDetector, logger: Logger
    ) -> None:
        self.logger: Logger = logger
        self.logger.info("Initializing calibration...")
        self.cfg: DictConfig = cfg
        self.detector = detector
        # self.features_gt: List[List[Features]] = self.__collect_features_gt()
        # self.time_instants: int = len(self.features_gt)
        # self.n_cameras: int = len(self.features_gt[0])
        self.time_instants: int = 10
        self.n_cameras: int = 3
        self.obj = obj.to(self.cfg.calibration.device)
        self.objects = self.__get_objects()
        self.cameras: List[Camera_cv] = self.__get_cameras()

    def run(self) -> None:
        self.logger.info("Calibrating...")

        params = self.__collect_parameters()
        params_mask: List[torch.Tensor] = self.__collect_masks()
        optimizer = instantiate(self.cfg.optimizer, params=params)
        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)

        progress_bar = tqdm(range(self.cfg.calibration.iterations), desc="Iteration: ")
        for _ in progress_bar:

            # for each time instant
            loss_total = 0
            for time_id in range(self.time_instants):
                x, y = self.__get_xy(time_id)
                loss = self.__loss(x, y)
                loss_total += loss.item()
                loss.backward()

            self.__filter_gradients(params, params_mask)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({"loss": loss_total})

    def __get_objects(self) -> List[Object]:
        return [
            self.obj.clone(same_pose=False, same_relative_poses=True)
            for time_id in range(self.time_instants)
        ]
        pass

    def __loss(self, f_hat: torch.Tensor, f_gt: torch.Tensor) -> torch.Tensor:
        criterion = instantiate(self.cfg.loss)
        return criterion(f_hat, f_gt)

    def __collect_features_gt(self) -> List[List[Features]]:
        features = []
        coll_data = Collector.load(self.cfg.paths.collection_dir, in_ram=False)
        for images in coll_data.raw_images():
            features.append(self.detector.detect_features(images))
        return features

    def __get_xy(self, time_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        points_gt_list = []
        points_hat_list = []
        obj = self.objects[time_id]
        ids_list_hat = obj.ids()
        points3D_list_hat = obj.points()
        features_list = self.features_gt[time_id]
        for cam_id, f in enumerate(features_list):
            ids_list_gt = f.ids()
            points_list_gt = f.points()
            for board_id in range(len(ids_list_gt)):
                ids_gt = ids_list_gt[board_id]
                points_gt = points_list_gt[board_id]
                ids_hat = ids_list_hat[board_id]
                points3D_hat = points3D_list_hat[board_id]
                points3D_hat = points3D_hat[ids_hat == ids_gt]
                # TODO utils ema
                points_hat_undistort = self.cameras[cam_id].project_points(points3D_hat)
                points_hat = self.cameras[cam_id].distort(points_hat_undistort)
                points_gt_list.append(points_gt)
                points_hat_list.append(points_hat)
        return torch.cat(points_gt_list, dim=0), torch.cat(points_hat_list, dim=0)

    def __get_cameras(self) -> List[Camera_cv]:
        cameras = []
        for _ in range(self.n_cameras):
            cameras.append(Camera_cv(device=self.cfg.calibration.device))
        return cameras

    def __collect_masks(self) -> List[torch.Tensor]:
        masks = []
        for _ in range(self.time_instants):
            masks.append(torch.Tensor(self.cfg.calibration.objects_pose_fix.eul))
            masks.append(torch.Tensor(self.cfg.calibration.objects_pose_fix.position))

        for _ in range(self.n_cameras):
            masks.append(torch.ones(3))  # euler
            masks.append(torch.ones(3))  # position
            masks.append(torch.ones(4))  # fx,fy,cx,cy
            masks.append(torch.ones(5))  # dist params

        return masks

    def __collect_parameters(self) -> List[torch.Tensor]:
        # collect object parameters
        params = []
        for time_id in range(self.time_instants):
            obj = self.objects[time_id]
            pose = obj.pose
            params.append(pose.euler.e)  # euler
            params.append(pose.position)  # position
        for o in params:
            o.requires_grad = True

        # collect cam parameters
        for cam_id in range(self.n_cameras):
            cam = self.cameras[cam_id]
            # extrinsics
            params.append(cam.pose.euler.e)  # euler
            params.append(cam.pose.position)  # position
            # intrinsics
            params.append(cam.intr.K_params)
            params.append(cam.intr.D_params)
        return params

    def __filter_gradients(self, params, params_mask) -> None:
        for i, p in enumerate(params):
            p.grad *= params_mask[i]

    def save(self) -> None:
        process = subprocess.run(
            [
                "blenderproc",
                "run",
                "/home/manu/Desktop/repositories/refactored/opticalib/src/blender_saver.py",
            ]
        )

        # blender_save(
        #     self.cfg.paths.calib_results_dir, self.objects, self.cameras, self.logger
        # )

    def load(self) -> bool:
        return True
