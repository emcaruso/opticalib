import torch
from tqdm import tqdm
from typing import List, Dict
from logging import Logger
from omegaconf import DictConfig
from hydra.utils import instantiate
from utils_ema.charuco import Image
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

        self.optimizer = instantiate(self.cfg.params.optimizer, self.params)
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
            # print(self.scene.objects.pose.position[24,0,...].grad)
            # print(self.scene.objects.pose.orientation.params[24,0,...])
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
        self.__update_params()
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
            latest_show = ((it == -1) and self.cfg.test.calib_show_last)
            if latest_show:
                self.logger.info("Press a button after selecting displaying images to continue.")
            Image.show_multiple_images(images, wk = int(not latest_show) )


            # # version with flat masked
            # x, y = self.scene.get_xy_flat_masked(pixel_unit=True)
            # r = self.scene.cameras.intr.resolution[0,0,0,...].type(torch.int64)
            # image = Image(torch.ones([r[0].item(), r[1].item(), 3]))
            # image.draw_circles(x, color=(0,0,255), radius = 7, thickness=4)
            # image.draw_circles(y, color=(255,0,0), radius = 7)
            # image.draw_lines(x, y, color=(0,255,0))
            # latest_show = ((it == -1) and self.cfg.test.calib_show_last)
            # if latest_show:
            #     self.logger.info("Press a button after selecting displaying images to continue.")
            # image.show(wk = int(not latest_show))
            #
    def __mean_distance(self, f_hat: torch.Tensor, f_gt: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(f_hat-f_gt, dim=1)
        return torch.mean(dist)

    def __loss(self, f_hat: torch.Tensor, f_gt: torch.Tensor) -> torch.Tensor:

        # reprojection loss
        criterion = instantiate(self.cfg.params.loss)
        loss_reprojection = criterion(f_hat, f_gt)

        # loss_cos = self.__cos_loss(f_hat, f_gt)
        # return 1 * loss_reprojection + 10000 *loss_cos
        
        return loss_reprojection


    def __cos_loss(self, f_hat: torch.Tensor, f_gt: torch.Tensor) -> torch.Tensor:

        # direction loss
        f_dir_hat = (f_hat.unsqueeze(1) - f_hat.unsqueeze(1).transpose(-2, -1)).reshape(-1,2)
        f_dir_gt = (f_gt.unsqueeze(1) - f_gt.unsqueeze(1).transpose(-2, -1)).reshape(-1,2)


        f_dir_hat_normalized = f_dir_hat / torch.norm(f_dir_hat, dim=1, keepdim=True)
        f_dir_gt_normalized = f_dir_gt / torch.norm(f_dir_gt, dim=1, keepdim=True)

        loss_cos = 1 - torch.sum(f_dir_hat_normalized * f_dir_gt_normalized, dim=1)
        return torch.mean(loss_cos)


    def __regularization_step(self) -> torch.Tensor:
        if self.intr_K:
            loss_reg_list = []
            loss_reg_list.append(torch.abs(self.scene.cameras.intr.K_params[...,0] - self.scene.cameras.intr.K_params[...,1]))
            # loss_reg_list.append(torch.abs(cam.intr.K_params[2] - cam.intr.K_params[3]))
            loss_reg = self.cfg.reg_weight * torch.stack(loss_reg_list).sum()
            loss_reg.backward()


    def __update_params(self) -> None:
        with torch.no_grad():
            self.scene.cameras.intr.K_params.data = torch.abs(self.scene.cameras.intr.K_params).data
        self.scene.cameras.intr.update_intrinsics()

        # for quaternions
        self.scene.cameras.pose.orientation.normalize_data()
        self.scene.objects.pose.orientation.normalize_data()
        self.scene.objects.relative_poses.orientation.normalize_data()
        self.scene.world_pose.orientation.normalize_data()
        
        
    # TODO
    def __collect_parameters(self) -> List[Dict]:

        params = []
        masks = []

        # collect object poses
        if self.obj_pose:
            position = self.scene.objects.pose.position
            orientation = self.scene.objects.pose.orientation.params
            params.append({"params":position, "lr":0.001*self.cfg.lr})
            params.append({"params":orientation, "lr":1*self.cfg.lr})
            if "objects_pose_opt" in self.cfg.keys():
                position_mask = torch.zeros_like(position)
                position_mask[...,0] = int(self.cfg.objects_pose_opt.position[0])
                position_mask[...,1] = int(self.cfg.objects_pose_opt.position[1])
                position_mask[...,2] = int(self.cfg.objects_pose_opt.position[2])
                ori_mask = torch.ones_like(orientation)
                # ori_mask = torch.zeros_like(orientation)
                # ori_mask[...,0] = int(self.cfg.objects_pose_opt.eul[0])
                # ori_mask[...,1] = int(self.cfg.objects_pose_opt.eul[1])
                # ori_mask[...,2] = int(self.cfg.objects_pose_opt.eul[2])
                masks.append(position_mask)
                masks.append(ori_mask)
            else:
                masks.append(torch.ones_like(position))
                masks.append(torch.ones_like(orientation))
        
        # collect object relative
        if self.obj_rel:
            params.append({"params":self.scene.objects.relative_poses.position, "lr":0.001*self.cfg.lr})
            params.append({"params":self.scene.objects.relative_poses.orientation.params, "lr":1*self.cfg.lr})
            pos = torch.ones_like(self.scene.objects.relative_poses.position)
            ori = torch.ones_like(self.scene.objects.relative_poses.orientation.params)
            pos[:,:,0,...] = 0
            ori[:,:,0,...] = 0
            masks.append(pos)
            masks.append(ori)

        # collect extrinsics cam parameters
        if self.extr:
            params.append({"params":self.scene.cameras.pose.position, "lr":0.001*self.cfg.lr})
            params.append({"params":self.scene.cameras.pose.orientation.params, "lr":1*self.cfg.lr})
            masks.append(torch.ones_like(self.scene.cameras.pose.position))
            masks.append(torch.ones_like(self.scene.cameras.pose.orientation.params))

        # collect intrinsics
        if self.intr_K:
            params.append({"params":self.scene.cameras.intr.K_params, "lr":1*self.cfg.lr})
            masks.append(torch.ones_like(self.scene.cameras.intr.K_params))

        # collect distortion coefficients
        if self.intr_D:
            params.append({"params":self.scene.cameras.intr.D_params, "lr":10*self.cfg.lr})
            masks.append(torch.ones_like(self.scene.cameras.intr.D_params))

        # collect world rotation
        if self.world_rot:
            p = self.scene.world_pose
            params.append({"params":p.orientation.params, "lr":1*self.cfg.lr})
            masks.append(torch.zeros_like(p.orientation.params))
            # mask = torch.tensor(self.cfg.world_rotation.eul, device=p.device)[None,None,None]
            # masks.append(mask)
            # masks.append(torch.ones_like(p.orientation.params))

        for p in params:
            p["params"].requires_grad = True

        return params, masks

    def __update_gradients(self, params, params_mask) -> None:
        for i, p in enumerate(params):
            if p["params"][0].grad is not None:
                p["params"][0].grad *= params_mask[i]


