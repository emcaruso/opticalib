import torch
import cv2
import sys
import subprocess
import numpy as np

# from blender_saver import blender_save
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from logging import Logger
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate
from utils_ema.charuco import Image
from objects.object import Object, Features, ObjectDetector
from blender_saver import blender_save
from utils_ema.plot import plotter
from scene import Scene


class Optimizer:
    def __init__(
        self, cfg: DictConfig, scene: Scene, logger: Logger,
        intr_K: bool = True, intr_D: bool = True, extr: bool = True, obj_rel: bool = True, obj_pose: bool = True,
        n_features_min: int = 0,
    ) -> None:
        self.logger: Logger = logger
        self.cfg: DictConfig = cfg
        self.scene = scene

        self.intr_K = intr_K
        self.intr_D = intr_D
        self.extr = extr
        self.obj_rel = obj_rel
        self.obj_pose = obj_pose
        self.world_rot = cfg.world_rotation

        self.n_features_min = n_features_min

    def __early_stopping(self, loss: torch.Tensor) -> bool:
        if not hasattr(self, "loss"):
            self.loss = loss
            return False
        else:
            if torch.abs(loss-self.loss) < self.cfg.early_stopping_diff:
                return True
            else: 
                self.loss = loss
                return False

    def run(self) -> None:
        self.params, self.params_mask = self.__collect_parameters()

        self.optimizer = instantiate(self.cfg.params.optimizer, params=self.params, lr = self.cfg.lr)
        self.scheduler = instantiate(self.cfg.params.scheduler, optimizer=self.optimizer)
        progress_bar = tqdm(range(self.cfg.iterations), desc="Iteration: ")


        # training loop
        for it in progress_bar:

            self.__visualize(it)
            self.__update_params()

            # backward
            x, y = self.scene.get_xy_flat_masked(pixel_unit=True)
            loss = self.__loss(x, y)
            dist = self.__mean_distance(x, y)
            loss.backward()
            
            # update
            self.__regularization_step()
            self.__update_gradients(self.params, self.params_mask)
            self.optimizer.step()
        
            self.optimizer.zero_grad()
            self.scheduler.step()
            if self.__early_stopping(loss):
                break

            # progress bar update
            progress_bar.set_postfix({"average distance": f"{dist:9.3f}",
                                      "average loss": f"{loss:9.3f}"})

            # for visualization purposes
        if self.cfg.test.calib_show_realtime and self.cfg.iterations > 0:
            self.__visualize(-1)

    def __visualize(self, it):
        if (self.cfg.test.calib_show_realtime and it % self.cfg.test.calib_show_rate == 0 and self.cfg.test.calib_show_rate != -1) or it == -1:
            images = []
            x_tens, y_tens, mask = self.scene.get_xy(pixel_unit=True)
            for cam_id in range(self.scene.n_cameras):
                idxs = mask[:,cam_id,...].reshape(-1)
                x = x_tens[:,cam_id,...].reshape(-1,2)[idxs]
                y = y_tens[:,cam_id,...].reshape(-1,2)[idxs]
                r = self.scene.cameras.intr.resolution[0,cam_id,0,...].type(torch.int64)
                image = Image(torch.ones([r[0].item(), r[1].item(), 3]))
                image.draw_circles(x, color=(0,0,255), radius = 7, thickness=4)
                image.draw_circles(y, color=(255,0,0), radius = 7)
                image.draw_lines(x, y, color=(0,255,0))
                images.append(image)
            Image.show_multiple_images(images, wk=1)

    def __mean_distance(self, f_hat: torch.Tensor, f_gt: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.norm(f_hat-f_gt, dim=1))

    def __loss(self, f_hat: torch.Tensor, f_gt: torch.Tensor) -> torch.Tensor:
        criterion = instantiate(self.cfg.params.loss)
        return criterion(f_hat, f_gt)

    def __regularization_step(self) -> torch.Tensor:
        if self.intr_K:
            loss_reg_list = []
            loss_reg_list.append(torch.abs(self.scene.cameras.intr.K_params[...,0] - self.scene.cameras.intr.K_params[...,1]))
            # loss_reg_list.append(torch.abs(cam.intr.K_params[2] - cam.intr.K_params[3]))
            loss_reg = self.cfg.reg_weight * torch.stack(loss_reg_list).sum()
            loss_reg.backward()


    # def get_xy(self, time_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    #
    #     points_gt_list = []
    #     points_hat_list = []
    #     obj = self.scene.objects[time_id]
    #     ids_list_hat = obj.ids()
    #     points3D_list_hat = obj.points()
    #     features_list = self.scene.features_gt[time_id]
    #     for cam_id in range(self.scene.n_cameras):
    #         f = features_list[cam_id]
    #         ids_list_gt = f.ids
    #         points_list_gt = f.points
    #         cam = self.scene.cameras[cam_id] 
    #         cam.intr.update_intrinsics()
    #         for board_id in range(len(ids_list_gt)):
    #             ids_gt = ids_list_gt[board_id]
    #             if len(ids_gt) < self.n_features_min:
    #                 continue
    #             ids_hat = ids_list_hat[board_id]
    #             points_gt = points_list_gt[board_id]
    #             points3D_hat = points3D_list_hat[board_id]
    #             idxs = torch.where(torch.isin(ids_hat, ids_gt.squeeze(-1)))[0]
    #             points3D_hat = points3D_hat[idxs]
    #             points2D_hat_undistort = cam.project_points(points3D_hat, longtens=False, und=False)
    #             points_hat = cam.distort(points2D_hat_undistort)
    #             # points_hat = points2D_hat_undistort 
    #             points_gt_list.append(points_gt)
    #             points_hat_list.append(points_hat)
    #
    #     if len(points_hat_list) == 0:
    #         return torch.tensor(points_hat_list), torch.tensor(points_hat_list)
    #
    #     return torch.cat(points_hat_list, dim=0), torch.cat(points_gt_list, dim=0).squeeze(1)


    def __update_params(self) -> None:
        with torch.no_grad():
            self.scene.cameras.intr.K_params.data = torch.abs(self.scene.cameras.intr.K_params).data
            # print(self.scene.world_pose.euler.e.data)

            # if self.world_rot:
            #     world_transf = self.scene.world_pose
            #     new_pose = world_transf * self.scene.cameras.pose
            #     self.scene.cameras.pose.position.data = new_pose.position.data
            #     self.scene.cameras.pose.euler.e.data = new_pose.euler.e.data
            #     self.scene.world_pose.position.data *= 0
            #     self.scene.world_pose.euler.e.data *= 0
            
            
    # TODO
    def __collect_parameters(self) -> List[Dict]:

        params = []
        masks = []

        # collect object poses
        if self.obj_pose:
            position = self.scene.objects.pose.position
            euler = self.scene.objects.pose.euler.e
            params.append(position)
            params.append(euler)
            if "objects_pose_opt" in self.cfg.keys():
                position_mask = torch.zeros_like(position)
                position_mask[...,0] = int(self.cfg.objects_pose_opt.position[0])
                position_mask[...,1] = int(self.cfg.objects_pose_opt.position[1])
                position_mask[...,2] = int(self.cfg.objects_pose_opt.position[2])
                euler_mask = torch.zeros_like(euler)
                euler_mask[...,0] = int(self.cfg.objects_pose_opt.eul[0])
                euler_mask[...,1] = int(self.cfg.objects_pose_opt.eul[1])
                euler_mask[...,2] = int(self.cfg.objects_pose_opt.eul[2])
                masks.append(position_mask)
                masks.append(euler_mask)
            else:
                masks.append(torch.ones_like(position))
                masks.append(torch.ones_like(euler))
        
        # collect object relative
        if self.obj_rel:
            params.append(self.scene.objects.relative_poses.position)
            params.append(self.scene.objects.relative_poses.euler.e)
            masks.append(torch.ones_like(self.scene.objects.relative_poses.position))
            masks.append(torch.ones_like(self.scene.objects.relative_poses.euler.e))

        # collect extrinsics cam parameters
        if self.extr:
            params.append(self.scene.cameras.pose.position)
            params.append(self.scene.cameras.pose.euler.e)
            masks.append(torch.ones_like(self.scene.cameras.pose.position))
            masks.append(torch.ones_like(self.scene.cameras.pose.euler.e))

        # collect intrinsics
        if self.intr_K:
            params.append(self.scene.cameras.intr.K_params)
            masks.append(torch.ones_like(self.scene.cameras.intr.K_params))

        # collect distortion coefficients
        if self.intr_D:
            params.append(self.scene.cameras.intr.D_params)
            masks.append(torch.ones_like(self.scene.cameras.intr.D_params))

        # collect world rotation
        if self.world_rot:
            p = self.scene.world_pose
            params.append(p.euler.e)
            # masks.append(torch.zeros_like(p.euler.e))
            mask = torch.tensor(self.cfg.world_rotation.eul, device=p.device)[None,None,None]
            masks.append(mask)
            # masks.append(torch.ones_like(p.euler.e))

        for p in params:
            p.requires_grad = True

        return params, masks

    # def __collect_parameters(self) -> Tuple[List[torch.Tensor],List[torch.Tensor]]:
    #
    #     params = []
    #     masks = []
    #
    #     # collect object poses
    #     if self.obj_pose:
    #         for time_id in range(self.scene.time_instants):
    #             obj = self.scene.objects[time_id]
    #             pose = obj.pose
    #             params.append(pose.euler.e)  # euler
    #             params.append(pose.position)  # position
    #             if "objects_pose_opt" in self.cfg.keys():
    #                 masks.append(torch.tensor(self.cfg.objects_pose_opt.eul, device=self.cfg.device))
    #                 masks.append(torch.tensor(self.cfg.objects_pose_opt.position, device=self.cfg.device))
    #             else:
    #                 masks.append(torch.ones(3, device=self.cfg.device))
    #                 masks.append(torch.ones(3, device=self.cfg.device))
    #
    #     # collect object relative
    #     if self.obj_rel:
    #         for o in self.scene.objects:
    #             for r in o.relative_poses:
    #                 if not any(r.euler.e is t for t in params):
    #                     params.append(r.euler.e)  # euler
    #                     masks.append(torch.ones(3, device=self.cfg.device))
    #                 if not any(r.position is t for t in params):
    #                     params.append(r.position)  # position
    #                     masks.append(torch.ones(3, device=self.cfg.device))
    #
    #     # collect extrinsics cam parameters
    #     if self.extr:
    #         for cam_id in range(self.scene.n_cameras):
    #             cam = self.scene.cameras[cam_id]
    #             params.append(cam.pose.euler.e)  # euler
    #             params.append(cam.pose.position)  # position
    #             masks.append(torch.ones(3, device=self.cfg.device))
    #             masks.append(torch.ones(3, device=self.cfg.device))
    #
    #     # collect intrinsics
    #     if self.intr_K:
    #         for cam_id in range(self.scene.n_cameras):
    #             cam = self.scene.cameras[cam_id]
    #             params.append(cam.intr.K_params)
    #             masks.append(torch.ones(4, device=self.cfg.device))
    #
    #     # collect distortion coefficients
    #     if self.intr_D:
    #         for cam_id in range(self.scene.n_cameras):
    #             cam = self.scene.cameras[cam_id]
    #             params.append(cam.intr.D_params)
    #             masks.append(torch.ones(5, device=self.cfg.device))
    #
    #     # collect world rotation
    #     if self.world_rot:
    #         p = self.scene.world_pose
    #         params.append(p.euler.e)
    #         masks.append(torch.ones(3, device=self.cfg.device))
    #
    #
    #     for p in params:
    #         p.requires_grad = True
    #
    #     return params, masks

    def __update_gradients(self, params, params_mask) -> None:
        for i, p in enumerate(params):
            if p.grad is not None:
                p.grad *= params_mask[i]


