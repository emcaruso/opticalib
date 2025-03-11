from omegaconf import DictConfig
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
import itertools
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

        # init cameras extrinsics
        self.__init_cameras_extrinsics(scene)
        
        return scene

    def __get_scene(self) -> Scene:
        features_gt = self.__collect_features_gt()
        cameras = self.__get_cameras(features_gt)
        objects = self.__get_objects(features_gt)
        scene = Scene(self.cfg, self.logger, features_gt, cameras, objects)
        return scene

    def __precalibration(self, scene: Scene):

        relative_poses = {}

        points_3D, points_2D, infos = scene.get_3D_2D_points_foreach_cam(filter_by_n_features_min = True, numpy=True)

        for cam_id, p3D, p2D, info in zip(range(len(points_3D)), points_3D, points_2D, infos):
            
            # init relative poses to empty list
            for obj in scene.objects:
                obj.relative_poses = [ None for _ in range(obj.params.n_boards)]

            # if cam_id == 1: continue
            cam = scene.cameras[cam_id]
            K = cam.intr.K_pix.cpu().numpy()
            D = cam.intr.D_params.cpu().numpy()
            flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_FOCAL_LENGTH
            # flags = cv2.CALIB_USE_INTRINSIC_GUESS
            imageSize = tuple(cam.intr.resolution.numpy())
            res = cv2.calibrateCamera( objectPoints=p3D, imagePoints=p2D, imageSize=imageSize, cameraMatrix=K, distCoeffs=D, flags=flags)


            K_params = torch.FloatTensor([res[1][0,0], res[1][1,1], res[1][0,2], res[1][1,2]])
            K_params *= cam.intr.unit_pixel_ratio()
            D_params = torch.tensor(res[2]).squeeze()
            cam.intr.K_params = K_params
            cam.intr.D_params = D_params
            cam.intr.update_intrinsics()

            for i, info in enumerate(infos[cam_id]):

                pose = Pose.cvvecs2pose(res[3][i], res[4][i] * cam.intr.unit_pixel_ratio().item())
                time_id = info["time_id"]
                board_id = info["board_id"]
                scene.objects[time_id].relative_poses[board_id] = pose

                # print(info)
                # points = scene.objects[time_id].points()[board_id]
                # proj_points = cam.project_points(points, und=False)
                # proj_points = cam.distort(proj_points)
                # proj_points = torch.cat((proj_points, torch.zeros([proj_points.shape[0], 1], dtype=torch.float32)), dim=1)
                # plotter.reset()
                # plotter.plot_points(proj_points)
                # # plotter.plot_points(proj_points_cv2)
                # plotter.plot_points(torch.cat((torch.tensor(p2D[i]), torch.zeros([torch.tensor(p2D[i]).shape[0], 1], dtype=torch.float32)), dim=1))
                # plotter.show()
                #
            for time_id in range(scene.time_instants):

                # reference pose
                obj = scene.objects[time_id]
                pose_ref = obj.relative_poses[0]
                if pose_ref is None:
                    continue

                for board_id in range(1,scene.objects[0].params.n_boards):

                    # relative pose
                    pose_rel = obj.relative_poses[board_id]
                    if pose_rel is None:
                        continue

                    if board_id not in relative_poses.keys():
                        relative_poses[board_id] = []

                    relative_pose = pose_rel - pose_ref
                    relative_poses[board_id] += [relative_pose]


        # reinit objects with shared relative pose
        scene.objects = self.__get_objects(scene.features_gt, same_relative_poses=True)
        for board_id in relative_poses.keys():
            average_pose = Pose.average_poses(relative_poses[board_id])
            for obj in scene.objects:
                obj.relative_poses[board_id] = average_pose

        
    def __init_cameras_extrinsics(self, scene: Scene):

        import ipdb; ipdb.set_trace()


        points_3D, points_2D, infos = scene.get_3D_2D_points_foreach_cam(filter_by_n_features_min = True, numpy=True)

        # collect valid time ids
        valid_time_ids = []
        for time_id in range(scene.time_instants):
            comb = itertools.product(range(scene.n_cameras), range(scene.objects[0].params.n_boards))
            for cam_id, board_id in comb:
                l = len(scene.features_gt[time_id][cam_id].points[board_id])
                if l < scene.n_features_min:
                    continue
                valid_time_ids.append(time_id)

                    
        points = torch.cat(scene.objects[0].points(), dim=0).cpu().numpy()

        import ipdb; ipdb.set_trace()
        for board_id in range(1,scene.objects[0].params.n_boards):

            # solve pnp
            for cam_id in range(scene.n_cameras):

                res, rvec, tvec = cv2.solvePnP(points, points_np, K, D)


        pass

    def __intrinsics_init(self, scene: Scene):

        # precalibration
        self.__precalibration(scene)

        # for cam_id in range(len(scene.features_gt[0])):
        #     # align boards with cameras
        #     optim = Optimizer(cfg=self.cfg, scene=scene, logger=self.logger, cam_id=cam_id,
        #                         intr_K=False, intr_D=False, extr=False, obj_rel=True, obj_pose=False, n_features_min=scene.n_features_min)
        #                         # intr_K=False, intr_D=False, extr=False, obj_rel=True, obj_pose=False)
        #     optim.run()
        #
        #     import ipdb; ipdb.set_trace()
        

        #     self.logger.info(f"Calibrating intrinsics of camera {cam_id}")
        #
        #     # init relative poses
        #     for board_id in range(scene.objects[0].params.n_boards):
        #
        #         for time_id in range(len(self.features_gt)):
        #
        #             board = scene.objects[time_id]
        #             features = self.features_gt[time_id][cam_id]
        #             if len(features.points[board_id]) >= self.n_features_min:
        #
        #                 pose = board.get_pose_from_PnP(features, board_id, self.cameras[cam_id])
        #                 scene.objects[time_id].relative_poses[board_id] = pose
        #
                
            


        # # for obj in scene.objects:
        # #     obj.pose = Pose(position=torch.tensor([0,0,-1], dtype=torch.float32), euler=eul(torch.zeros(3, dtype=torch.float32)))
        #
        
        # import ipdb; ipdb.set_trace()
        #
        # # align boards + intrinsics
        # optim = Optimizer(cfg=self.cfg, scene=scene, logger=self.logger, cam_id=cam_id,
        #                 intr_K=True, intr_D=True, extr=False, obj_rel=True, obj_pose=True)
        # optim.run()
        

    def __get_cameras(self, features_gt) -> List[Camera_cv]:
        cameras = []
        n_cameras = len(features_gt[0])
        for i in range(n_cameras):

            # get resolution
            resolution = CollectorLoader.resolutions[i]
            info = CollectorLoader.load_info(self.cfg.paths.collection_dir)
            sensor_size = torch.FloatTensor(info[f"cam_{i:03d}"].SensorSize)*1e-3

            # prior intrinsics
            f = self.cfg.calibration.focal_length_prior
            c = sensor_size/2
            K=torch.FloatTensor([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
            D = torch.zeros(5)

            intr = Intrinsics(K=K, D=D, resolution=resolution, sensor_size=sensor_size)
            pose = Pose(position=torch.tensor([0,0,0], dtype=torch.float32), euler=eul(torch.zeros(3, dtype=torch.float32)))
            cameras.append(Camera_cv(device=self.cfg.calibration.device, intrinsics=intr, pose=pose))
        return cameras

    def __get_objects(self, features_gt, same_relative_poses = True) -> List[Object]:
        time_instants = len(features_gt)
        objects = [
            self.obj.clone(same_pose=False, same_relative_poses=same_relative_poses)
            for _ in range(time_instants)
        ]
        return objects

    def __collect_features_gt(self) -> List[List[Features]]:
        self.coll_data = CollectorLoader.load_images(self.cfg.paths.collection_dir, in_ram=self.cfg.collect.in_ram, raw=True)
        next(self.coll_data) # initialize static members
        if not Path(self.cfg.paths.features_file).exists():
            features = []
            self.logger.info("Collecting features...")
            progress_bar = tqdm(self.coll_data, desc="Collecting features", total = CollectorLoader.n_images)
            for images in progress_bar:
                features.append(self.detector.detect_features(images, device=self.cfg.calibration.device))
            # save custom object with pickle
            np.save(self.cfg.paths.features_file, features, allow_pickle=True)

        else:
            features = np.load(self.cfg.paths.features_file, allow_pickle=True)
            for feat in features:
                for f in feat:
                    f.to(self.cfg.calibration.device)
        return features
