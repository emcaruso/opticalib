from omegaconf import DictConfig
from collections import defaultdict
import cv2
from logging import Logger
from scene import Scene
from objects.object import Object, Features, ObjectDetector
from typing import List
from sensorflow.src.collector import CollectorLoader
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from optimizer import Optimizer
from utils_ema.camera_cv import Camera_cv, Intrinsics
from utils_ema.geometry_pose import Pose
from utils_ema.geometry_euler import eul
from utils_ema.plot import plotter


class Initializer():
    def __init__(self, cfg: DictConfig, logger: Logger, detector: ObjectDetector,  obj: Object):
        self.cfg = cfg
        self.logger = logger
        self.detector = detector
        self.obj = obj

    def initialize(self):

        # get scene
        scene = self.__get_scene()

        # init board relative poses + intrinsics (fixed focal length)
        self.__precalibration(scene)
        self.logger.info("Pre-calibrated intrinsics (fixed focal length), and relative poses")

        # init cameras extrinsics
        self.__init_cameras_extrinsics(scene)
        self.logger.info("Initialize camera extrinsics")

        # optimize to get object poses
        self.logger.info("Pre-calibrate object poses")
        optimizer = Optimizer(cfg=self.cfg.calibration.precalib_params, scene=scene, logger=self.logger,
                            intr_K=False, intr_D=False, extr=False, obj_rel=False, obj_pose=True,
                            n_features_min=scene.n_features_min,
                             )
        optimizer.run()
        
        return scene

    def __get_scene(self) -> Scene:
        features_gt = self.__collect_features_gt()
        cameras = self.__get_cameras(features_gt)
        objects = self.__get_objects(features_gt)
        scene = Scene(self.cfg, self.logger, features_gt, cameras, objects)
        return scene

    def __precalibration(self, scene: Scene):
        '''
        get object relative poses + camera intrinsics
        '''

        points_3D, points_2D = scene.get_3D_2D_points()
        n_board = points_2D.shape[2]

        for cam_id in range(scene.n_cameras):
            
            ratio = scene.cameras.intr.unit_pixel_ratio()[0,cam_id,0,0].item()
            p2D = points_2D[:,cam_id,...].reshape(-1, points_2D.shape[-2], 2) * (1/ratio)
            p3D = points_3D.reshape(-1, points_2D.shape[-2], 3) * (1/ratio)
            p3D_list = []
            p2D_list = []
            idxs_list = []
            for i in range(len(p3D)):
                idxs = p2D[i,...].sum(dim=-1)!=float('inf')
                if idxs.sum().item() > scene.n_features_min:
                    time_id = i // points_3D.shape[2]
                    board_id = i % points_3D.shape[2]

                    idxs_list.append((time_id, board_id))
                    p3D_list.append( p3D[i,...][idxs].cpu().numpy())
                    p2D_list.append( p2D[i,...][idxs].cpu().numpy())

            K = scene.cameras.intr.K_pix[0,cam_id,0,...].cpu().numpy()
            D = scene.cameras.intr.D_params[0,cam_id,0,...].cpu().numpy()
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
            # flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_FOCAL_LENGTH
            # flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
            # flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |  
            #         cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_FIX_FOCAL_LENGTH)
            imageSize = tuple(scene.cameras.intr.resolution[0,cam_id,0,:].int().cpu().numpy())
            # res = cv2.calibrateCamera( objectPoints=p3D_list, imagePoints=p2D_list, imageSize=imageSize, cameraMatrix=None, distCoeffs=None)
            res = cv2.calibrateCamera( objectPoints=p3D_list, imagePoints=p2D_list, imageSize=imageSize, cameraMatrix=K, distCoeffs=D, flags=flags)
            self.logger.info(f"Precalib error for camera {cam_id}: {res[0]}")

            scene.cameras.intr.K_params[0,cam_id,0,0] = res[1][0,0] * ratio
            scene.cameras.intr.K_params[0,cam_id,0,1] = res[1][1,1] * ratio
            scene.cameras.intr.K_params[0,cam_id,0,2] = res[1][0,2] * ratio
            scene.cameras.intr.K_params[0,cam_id,0,3] = res[1][1,2] * ratio
            scene.cameras.intr.D_params[0,cam_id,0,:] = torch.tensor(res[2]).squeeze()

            # compute relative poses
            poses_dict = defaultdict(lambda: defaultdict(dict))
            for i in range(len(idxs_list)):
                time_id, board_id = idxs_list[i]
                pose = Pose.cvvecs2pose(res[3][i], res[4][i] * ratio)
                poses_dict[time_id][board_id] = pose

            relative_poses = [ [] for _ in range(n_board) ]
            for time_id in poses_dict.keys():
                if 0 not in poses_dict[time_id].keys():
                    continue
                pose_ref = poses_dict[time_id][0]
                for board_id in range(n_board)[1:]:
                    if board_id not in poses_dict[time_id].keys():
                        continue
                    pose_rel = pose_ref - poses_dict[time_id][board_id]
                    relative_poses[board_id].append( pose_rel )

            for board_id in range(n_board)[1:]:
                avg_pose = Pose.average_poses(relative_poses[board_id])
                self.obj.relative_poses.position[0,0,board_id] = avg_pose.position
                self.obj.relative_poses.euler.e[0,0,board_id] = avg_pose.euler.e

        scene.cameras.intr.update_intrinsics()
    
    
    def __init_cameras_extrinsics(self, scene: Scene):


        mask = scene.features_gt!=float('inf')
        mask = (mask.sum(dim=-2)[...,0] >= scene.n_features_min)
        valid = mask.any(dim=1).all(dim=1)
        scores = mask.reshape(scene.time_instants, -1).sum(dim=-1)
        scores = scores * valid
        best_time_ids = torch.where(scores == scores.max())[0]
        best_score = scores[best_time_ids[0]].item()

        if best_score == 0:
            raise Exception("No valid time_id found to initialize cameras extrinsics")

        points_3D, points_2D = scene.get_3D_2D_points()
        repr_errors = []
        rvecs = []
        tvecs = []
        for best_time_id in best_time_ids:
            repr_sum = 0
            rvecs.append([]); tvecs.append([])
            for cam_id in range(scene.n_cameras):
                ratio = scene.cameras.intr.pixel_unit_ratio()[0,cam_id,0,0].item()
                p3D = points_3D[best_time_id, 0, ...].reshape(-1,3) * ratio
                p2D = points_2D[best_time_id, cam_id, ...].reshape(-1,2) * ratio
                idxs = (p2D!=float('inf')).any(dim=-1)
                p3D = p3D[idxs]
                p2D = p2D[idxs]
                K = scene.cameras.intr.K_pix[0,cam_id,0].cpu().numpy()
                D = scene.cameras.intr.D_params[0,cam_id,0].cpu().numpy()
                res, rvec, tvec = cv2.solvePnP(p3D.cpu().numpy(), p2D.cpu().numpy(),K , D)
                _, rvec_list, tvec_list, repr_list = cv2.solvePnPGeneric(p3D.cpu().numpy(), p2D.cpu().numpy(),K , D)
                idx_best = [np.array_equal(a, tvec) for a in tvec_list].index(True)
                repr = repr_list[idx_best].item()
                repr_sum += repr
                rvecs[-1].append(rvec)
                tvecs[-1].append(tvec)

            repr_errors.append(repr_sum/scene.n_cameras)
        
        min_repr_error = min(repr_errors)
        best_time_id = best_time_ids[repr_errors.index(min_repr_error)].item()
        idx = torch.where(best_time_ids == best_time_id)[0].item()
        self.logger.info(f"Mean reprojection error during extrinsics initialization: {min_repr_error}")
        for cam_id in range(scene.n_cameras):
            ratio = scene.cameras.intr.pixel_unit_ratio()[0,cam_id,0,0].item()
            rvec = rvecs[ idx ][ cam_id ]
            tvec = tvecs[ idx ][ cam_id ] * (1/ratio)

            cam_pose = Pose.cvvecs2pose(rvec, tvec)
            cam_pose.invert()
            scene.cameras.pose.position[0,cam_id,0] = cam_pose.position
            scene.cameras.pose.euler.e[0,cam_id,0] = cam_pose.euler.e
            # plot for test purposes
            if self.cfg.calibration.test.init_show:
                plotter.plot_pose(cam_pose)
                plotter.plot_points(points_3D[best_time_id, 0, ...].reshape(-1,3))
        if self.cfg.calibration.test.init_show:
            plotter.show()



        

    def __get_cameras(self, features_gt) -> Camera_cv:
        n_cameras = len(features_gt[0])
        K = torch.zeros([1,n_cameras,1,3,3])
        D = torch.zeros([1,n_cameras,1,5])
        resolution = torch.zeros([1,n_cameras,1,2])
        for i in range(n_cameras):

            # get resolution
            resolution[0,i,0,...] = CollectorLoader.resolutions[i]
            info = CollectorLoader.load_info(self.cfg.paths.collection_dir)
            pixel_size = torch.FloatTensor(info[f"cam_{i:03d}"].PixelSizeMicrometers)*1e-6
            crop_offset = torch.FloatTensor(info[f"cam_{i:03d}"]["crop_offset"])
            resolution_native = torch.FloatTensor(info[f"cam_{i:03d}"]["resolution_native"])

            # handle crop influence on principal point
            sensor_size = resolution*pixel_size
            if torch.all(CollectorLoader.resolutions[i].int()==resolution_native.int()).item():
                crop_offset = torch.zeros_like(crop_offset)

            # prior intrinsics
            f = self.cfg.calibration.focal_length_prior
            c = (resolution_native/2 - crop_offset)*pixel_size
            K[0,i,0,...] = torch.FloatTensor([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])

        intr = Intrinsics(K=K, D=D, resolution=resolution, sensor_size=sensor_size)

        position = torch.zeros([1,n_cameras,1,3], dtype=torch.float32)
        euler = eul(torch.zeros([1,n_cameras,1,3], dtype=torch.float32))
        pose = Pose(position=position, euler=euler)
        cameras = Camera_cv(device=self.cfg.calibration.device, intrinsics=intr, pose=pose)
        return cameras

    def __get_objects(self, features_gt) -> List[Object]:
        time_instants = len(features_gt)
        self.obj.pose.position = self.obj.pose.position.repeat(time_instants, 1, 1, 1)
        self.obj.pose.euler.e = self.obj.pose.euler.e.repeat(time_instants, 1, 1, 1)
        return self.obj

    def __collect_features_gt(self) -> List[List[Features]]:
        self.coll_data = CollectorLoader.load_images(self.cfg.paths.collection_dir, in_ram=self.cfg.collect.in_ram, raw=True)
        next(self.coll_data) # initialize static members


        if not Path(self.cfg.paths.features_file).exists():
            n_time_ids = CollectorLoader.n_images
            n_cam_ids = CollectorLoader.n_cams
            n_boards = self.obj.params.n_boards
            n_corners = self.obj.params.n_corners
            points_tensor = torch.full([n_time_ids, n_cam_ids, n_boards, n_corners, 2], float('inf'), dtype=torch.float32, device=self.cfg.calibration.device)
            self.logger.info("Collecting features...")
            for time_id, images in enumerate(tqdm(self.coll_data, total = CollectorLoader.n_images)):
                for cam_id in range(n_cam_ids):
                    features = self.detector.detect_features(images, device=self.cfg.calibration.device)[cam_id]
                    points_list = features.points
                    ids_list = features.ids
                    for board_id in range(n_boards):
                        points = points_list[board_id].squeeze(1)
                        ids = ids_list[board_id] - board_id * n_corners
                        points_tensor[time_id, cam_id, board_id, ids, :] = points.unsqueeze(1).type(torch.float32)
            torch.save( points_tensor, self.cfg.paths.features_file)
            return points_tensor 
        else:
            points_tensor = torch.load(self.cfg.paths.features_file)
            points_tensor = points_tensor.to(self.cfg.calibration.device)
            return points_tensor
