import numpy as np
import shutil
from tqdm import tqdm
import torch
import os
from objects.object import Object, Features
from pathlib import Path
from omegaconf import DictConfig
from logging import Logger
from typing import List, Tuple
from utils_ema.camera_cv import Camera_cv, Intrinsics
from utils_ema.geometry_pose import Pose
from utils_ema.image import Image
from utils_ema.geometry_euler import eul
from utils_ema.geometry_quaternion import Quat


class Scene():

    def __init__(self, cfg: DictConfig, logger: Logger, features_gt: List[List[Features]], cameras: Camera_cv, objects: List[Object]):
        self.cfg = cfg
        self.logger = logger
        self.features_gt = features_gt
        self.cameras = cameras
        self.objects = objects
        self.world_pose = self.__get_world_pose()
        self.n_cameras = len(self.features_gt[0])
        self.time_instants = len(self.features_gt)
        self.n_features_min = int(self.cfg.calibration.percentage_points_min*self.objects.params.n_corners)

    def __get_world_pose(self):
        n_dims = len(self.objects.pose.position.shape)

        # quaternions
        q_params = torch.zeros([1 for _ in range(n_dims)[1:]]+[4], dtype=torch.float32)
        q_params[..., 0] = 1
        o = Quat(q_params)
        
        # # euler
        # o = eul(torch.zeros([1 for _ in range(n_dims)[1:]]+[3], dtype=torch.float32))
        
        position = torch.zeros([1 for _ in range(n_dims)[1:]]+[3], dtype=torch.float32) 
        pose = Pose(orientation = o, position = position, device=self.cameras.device)
        return pose
        

    def get_xy_flat_masked(self, pixel_unit=False):
        p3D_tens, p2D_tens = self.get_3D_2D_points()
        # p2D_hat_und = self.cameras.project_points(p3D_tens, longtens=False, und=False )
        p2D_hat_und = self.cameras.project_points(p3D_tens, longtens=False, und=False, transform_cam_pose=self.world_pose)
        p2D_hat_tens = self.cameras.distort(p2D_hat_und)
        mask = ((p2D_tens!=float('inf')).all(dim=-1).view(-1))

        ratio = self.cameras.intr.pixel_unit_ratio()[...,None]
        if pixel_unit:
            p2D_scaled = p2D_tens * ratio
            p2D_hat_scaled = p2D_hat_tens
        else:
            p2D_scaled = p2D_tens
            p2D_hat_scaled = p2D_hat_tens * (1/ratio)

        p2D_hat = p2D_hat_scaled.reshape(-1, 2)[mask]
        p2D = p2D_scaled.reshape(-1, 2)[mask]

        if pixel_unit:
            return p2D_hat, p2D
        else:
            return p2D, p2D_hat

    def get_xy(self, pixel_unit=False) -> Tuple[torch.Tensor, torch.Tensor]:
        p3D, p2D = self.get_3D_2D_points()
        # p2D_hat_und = self.cameras.project_points(p3D, longtens=False, und=False)
        p2D_hat_und = self.cameras.project_points(p3D, longtens=False, und=False, transform_cam_pose=self.world_pose)
        p2D_hat = self.cameras.distort(p2D_hat_und)
        mask = (p2D!=float('inf')).any(dim=-1)
        ratio = self.cameras.intr.pixel_unit_ratio()[...,None]
        if pixel_unit:
            return p2D * ratio, p2D_hat, mask
        return p2D, p2D_hat * (1/ratio), mask

    def get_3D_2D_points(self):
        ratio = self.cameras.intr.pixel_unit_ratio()[...,None]
        p3D = self.objects.points(transform_world_rot=self.world_pose.rotation())
        # p3D = self.objects.points()
        p2D = self.features_gt * (1/ratio)
        return p3D, p2D

    def scene_postprocess_and_save_data(self):

        # apply world rotation
        self.objects.pose.euler = self.objects.pose.euler.rot2eul(self.world_pose.rotation().transpose(-2,-1) @ self.objects.pose.rotation())
        self.cameras.pose = self.world_pose.get_inverse_pose() * self.cameras.pose

        # save cameras pose
        with torch.no_grad():
            # save K, D, position, euler
            os.makedirs(self.cfg.paths.calib_results_dir, exist_ok=True)
            np.save(Path(self.cfg.paths.calib_results_dir) / "K.npy", self.cameras.intr.K.detach().cpu().numpy())
            np.save(Path(self.cfg.paths.calib_results_dir) / "D.npy", self.cameras.intr.D_params.detach().cpu().numpy())
            np.save(Path(self.cfg.paths.calib_results_dir) / "position.npy", self.cameras.pose.position.detach().cpu().numpy())
            np.save(Path(self.cfg.paths.calib_results_dir) / "euler.npy", self.cameras.pose.euler.e.detach().cpu().numpy())

        # from single to multiple cameras
        with torch.no_grad():
            cams = []
            for cam_id in range(self.n_cameras):
                K = self.cameras.intr.K.reshape(-1,3,3)[cam_id]
                D = self.cameras.intr.D_params.reshape(-1,5)[cam_id]
                resolution = self.cameras.intr.resolution.reshape(-1,2)[cam_id]
                sensor_size = self.cameras.intr.sensor_size.reshape(-1,2)[cam_id]
                intr = Intrinsics(K=K.cpu(), D=D.cpu(), resolution=resolution.cpu(), sensor_size=sensor_size.cpu())
                position = self.cameras.pose.position.reshape(-1,3)[cam_id]
                euler = self.cameras.pose.euler.e.reshape(-1,3)[cam_id]
                pose = Pose(position=position, euler=eul(euler))
                cam = Camera_cv(intrinsics=intr, pose=pose, name=f"cam_{cam_id:03d}")
                cams.append(cam)
            self.cameras = cams

        # undistort and save images
        undistort_dir = Path(self.cfg.paths.calib_results_dir) / "undist_images"
        shutil.rmtree(str(undistort_dir), ignore_errors=True)
        os.makedirs(str(undistort_dir), exist_ok=True)
        for cam_id, cam in enumerate(self.cameras):
            for image_id in tqdm(range(self.time_instants), desc=f"Undistorting images for cam { cam_id }"):
                path = Path(self.cfg.paths.collection_dir) / "raw" / f"cam_{cam_id:03d}" / f"{image_id:03d}.png"
                if path.exists():
                    img_dst = Image.from_path(str(path))

                    # # Test undist
                    # import cv2
                    # from objects.object import CharucoDetector
                    # cam.intr.D_params = torch.tensor([-1.9, 1.9, 0, 0, -1.9])
                    # cam.intr.compute_undistortion_map()
                    # img_und = cam.intr.undistort_image(img_dst)
                    # detector = CharucoDetector(params = self.objects.params)
                    # p2D_und = detector.detect_features([img_und])[0].points[1]
                    # p2D_und = p2D_und.reshape(-1, 2)
                    # p2D_dst = cam.distort(p2D_und.cpu())
                    # img_und = img_und.draw_circles(p2D_und)
                    # img_dst = img_dst.draw_circles(p2D_dst)
                    # cv2.destroyAllWindows()
                    # img_und.show(img_name="dist")
                    # img_dst.show(img_name="ori")
                    # cv2.waitKey(0)
                    
                    img_und = cam.intr.undistort_image(img_dst)
                    img_und.save(undistort_dir / f"cam_{cam_id:03d}" / f"{image_id:03d}.png")


