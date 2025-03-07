import torch
import subprocess
import numpy as np

# from blender_saver import blender_save
from tqdm import tqdm
from typing import List, Tuple, Optional
from logging import Logger
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate
from objects.object import Object, Features, ObjectDetector
from blender_saver import blender_save
from utils_ema.plot import plotter
from scene import Scene


class Optimizer:
    def __init__(
        self, cfg: DictConfig, scene: Scene, logger: Logger, cam_id: Optional[int] = None,
        intr_K: bool = True, intr_D: bool = True, extr: bool = True, obj_rel: bool = True, obj_pose: bool = True
    ) -> None:
        self.logger: Logger = logger
        self.cfg: DictConfig = cfg
        self.scene = scene
        self.camera_ids = self.__get_camera_ids(cam_id)

        self.intr_K = intr_K
        self.intr_D = intr_D
        self.extr = extr
        self.obj_rel = obj_rel
        self.obj_pose = obj_pose

    def __get_camera_ids(self, cam_id: Optional[int]) -> List[int]:
        if cam_id is None:
            return [i for i in range(len(self.scene.cameras))]
        else:
            return [cam_id]

    def closure(self) -> None:
        self.optimizer.zero_grad()
        self.__update_params()
        self.scheduler.step()

        # for each time instant
        loss_total = []
        for time_id in range(self.scene.time_instants)[-1:]:
        # for time_id in range(self.scene.time_instants):
            x, y = self.__get_xy(time_id)
            if len(x) > 100:
                loss = self.__loss(x, y)
                loss_total.append( loss )
        loss = torch.mean(torch.stack(loss_total))
        loss.backward()

        # self.__regularization_step()

        self.__filter_gradients(self.params, self.params_mask)

        # optimizer.step()
        # print(self.scene.cameras[0].intr.K_params)
        # print(self.scene.objects[-1].relative_poses[0].position)
        # print(self.scene.objects[-1].relative_poses[0].euler.e)
        # print(scheduler.get_last_lr())
        
        return loss

    def run(self) -> None:
        self.params, self.params_mask = self.__collect_parameters()

        self.optimizer = instantiate(self.cfg.optimizer, params=self.params)
        self.scheduler = instantiate(self.cfg.scheduler, optimizer=self.optimizer)
        progress_bar = tqdm(range(self.cfg.calibration.iterations), desc="Iteration: ")
        for _ in progress_bar:
            loss = self.optimizer.step(self.closure)
            progress_bar.set_postfix({"loss": f"{loss:9.3f}"})


        x, y = self.__get_xy(-1)
        x = torch.cat([x,torch.zeros([x.shape[0],1])], dim=1)
        y = torch.cat([y,torch.zeros([y.shape[0],1])], dim=1)
        plotter.plot_points(x)
        plotter.plot_points(y)
        plotter.show()
        import ipdb; ipdb.set_trace()
        
        
    def __loss(self, f_hat: torch.Tensor, f_gt: torch.Tensor) -> torch.Tensor:
        criterion = instantiate(self.cfg.loss)
        return criterion(f_hat, f_gt)

    def __regularization_step(self) -> torch.Tensor:
        if self.intr_K:
            loss_reg_list = []
            for cam in self.scene.cameras:
                loss_reg_list.append(torch.abs(cam.intr.K_params[0] - cam.intr.K_params[1]))
                loss_reg_list.append(torch.abs(cam.intr.K_params[2] - cam.intr.K_params[3]))
            loss_reg = self.cfg.calibration.reg_weight * torch.stack(loss_reg_list).sum()
            loss_reg.backward()


    def __get_xy(self, time_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        points_gt_list = []
        points_hat_list = []
        obj = self.scene.objects[time_id]
        ids_list_hat = obj.ids()
        points3D_list_hat = obj.points()
        features_list = self.scene.features_gt[time_id]
        for cam_id in self.camera_ids:
            f = features_list[cam_id]
            ids_list_gt = f.ids
            points_list_gt = f.points
            cam = self.scene.cameras[cam_id] 
            cam.intr.update_intrinsics()
            for board_id in range(len(ids_list_gt)):
                ids_gt = ids_list_gt[board_id]
                ids_hat = ids_list_hat[board_id]
                points_gt = points_list_gt[board_id]
                points3D_hat = points3D_list_hat[board_id]
                idxs = torch.where(torch.isin(ids_hat, ids_gt.squeeze(-1)))[0]
                points3D_hat = points3D_hat[idxs]
                points2D_hat_undistort = cam.project_points(points3D_hat, longtens=False, und=False)
                points_hat = cam.distort(points2D_hat_undistort)
                # points_hat = points2D_hat_undistort 
                points_gt_list.append(points_gt)
                points_hat_list.append(points_hat)
        return torch.cat(points_hat_list, dim=0), torch.cat(points_gt_list, dim=0).squeeze(1)


    def __update_params(self) -> None:
        for cam_id in self.camera_ids:
            cam = self.scene.cameras[cam_id]
            cam.intr.K_params.data = torch.abs(cam.intr.K_params).data


    def __collect_parameters(self) -> Tuple[List[torch.Tensor],List[torch.Tensor]]:

        params = []
        masks = []

        # collect object poses
        if self.obj_pose:
            for time_id in range(self.scene.time_instants):
                obj = self.scene.objects[time_id]
                pose = obj.pose
                params.append(pose.euler.e)  # euler
                params.append(pose.position)  # position
                masks.append(torch.tensor(self.cfg.calibration.objects_pose_opt.eul, device=self.cfg.calibration.device))
                masks.append(torch.tensor(self.cfg.calibration.objects_pose_opt.position, device=self.cfg.calibration.device))
        
        # collect object relative
        if self.obj_rel:
            for o in self.scene.objects:
                for r in o.relative_poses:
                    if not any(r.euler.e is t for t in params):
                        params.append(r.euler.e)  # euler
                        masks.append(torch.ones(3, device=self.cfg.calibration.device))
                    if not any(r.position is t for t in params):
                        params.append(r.position)  # position
                        masks.append(torch.ones(3, device=self.cfg.calibration.device))

        # collect extrinsics cam parameters
        if self.extr:
            for cam_id in self.camera_ids:
                cam = self.scene.cameras[cam_id]
                params.append(cam.pose.euler.e)  # euler
                params.append(cam.pose.position)  # position
                masks.append(torch.ones(3, device=self.cfg.calibration.device))
                masks.append(torch.ones(3, device=self.cfg.calibration.device))

        # collect intrinsics
        if self.intr_K:
            for cam_id in self.camera_ids:
                cam = self.scene.cameras[cam_id]
                params.append(cam.intr.K_params)
                masks.append(torch.zeros(4, device=self.cfg.calibration.device))

        # collect distortion coefficients
        if self.intr_D:
            for cam_id in self.camera_ids:
                cam = self.scene.cameras[cam_id]
                params.append(cam.intr.D_params)
                masks.append(torch.ones(5, device=self.cfg.calibration.device))

        for p in params:
            p.requires_grad = True

        return params, masks

    def __filter_gradients(self, params, params_mask) -> None:
        for i, p in enumerate(params):
            if p.grad is not None:
                p.grad *= params_mask[i]

